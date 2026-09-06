clear;
clc;
close all;

%% 1. SYSTEM PARAMETERS 
c = 3e8;
fc = 77e9;
lambda = c/fc;
delta_R = 0.2;
B = c/(2*delta_R);
Fs = 60e6;
N_samples = 2350;
Tc = N_samples/Fs;
S = B/Tc;
N_chirps = 64; 
R_nyquist = (c * Fs/2) / (2 * S);
v_max_unambig = lambda / (4 * Tc);

fprintf('System Nyquist Unambiguous Range: %.1f m\n', R_nyquist);
fprintf('System Max Unambiguous Velocity: +/-%.2f m/s\n', v_max_unambig);

%% 2. MULTI-TARGET 
target_Rs = [300.0, 600.0];       
target_vs = [-10.0, -40.0];        
num_targets = length(target_Rs);

for i = 1:num_targets
    fprintf('Target %d - True Range : %.1f m | True Velocity : %.2f m/s\n', i, target_Rs(i), target_vs(i));
end

%% 3. SIGNAL GENERATION 
tau_true = 2 * target_Rs / c;     
single_waveform = phased.FMCWWaveform( ...
    'SweepTime', Tc, ...
    'SweepBandwidth', B, ...
    'SampleRate', Fs, ...
    'SweepDirection', 'Up', ...
    'NumSweeps', 1);
tx_single = single_waveform();
tx = repmat(tx_single, N_chirps, 1);

rx_total = zeros(size(tx));
for m = 1:N_chirps
    t_slow = (m - 1) * Tc;
    rx_chirp_m = zeros(size(tx_single));
    for target_idx = 1:num_targets
        fd_true_m = 2 * target_vs(target_idx) / lambda;
        rx_single_target = delayseq(tx_single, tau_true(target_idx), Fs) * exp(1j * 2 * pi * fd_true_m * t_slow);
        rx_chirp_m = rx_chirp_m + rx_single_target;
    end
    idx_start = (m-1) * N_samples + 1;
    idx_end = m * N_samples;
    rx_total(idx_start:idx_end) = rx_chirp_m;
end
mix = tx .* conj(rx_total);

%% 4. 2D FFT 
n_fft_range = 4096;
n_fft_doppler = 128;
max_search_range = 1000.0;
max_k = floor(max_search_range / R_nyquist); 

rough_detected_ranges = zeros(1, num_targets);
rough_detected_velocities = zeros(1, num_targets);
residual_mix = mix;

range_win = hann(N_samples);
doppler_win = hann(N_chirps)';
fft_freq_axis = (0 : n_fft_range-1)' * (Fs / n_fft_range);
fft_range_axis = fft_freq_axis * c / (2 * S);
rough_velocity_axis = linspace(-50, 50, 51); 

for i = 1:num_targets
    Mix_matrix = reshape(residual_mix, N_samples, N_chirps);
    Mix_windowed = Mix_matrix .* range_win .* doppler_win;
    RDM_range = fft(Mix_windowed, n_fft_range, 1);
    RDM = abs(fftshift(fft(RDM_range, n_fft_doppler, 2), 2));
    
    [~, max_lin_idx] = max(RDM(:));
    [r_bin, ~] = ind2sub(size(RDM), max_lin_idx);
    
    R_alias = fft_range_axis(r_bin);
    candidate_ks = 0:max_k;
    candidate_ranges = R_alias + candidate_ks * R_nyquist;
    candidate_ranges = candidate_ranges(candidate_ranges <= max_search_range);
    
    best_joint_energy = -inf;
    best_r = R_alias;
    best_v = 0;
    
    for k_idx = 1:length(candidate_ranges)
        r_test = candidate_ranges(k_idx);
        tau_test = 2 * r_test / c;
        
        for v_idx = 1:length(rough_velocity_axis)
            v_test = rough_velocity_axis(v_idx);
            fd_test = 2 * v_test / lambda;
            
            rx_template_test = zeros(size(tx));
            for m_ch = 1:N_chirps
                t_slow = (m_ch - 1) * Tc;
                idx_s = (m_ch-1) * N_samples + 1;
                idx_e = m_ch * N_samples;
                rx_template_test(idx_s:idx_e) = delayseq(tx_single, tau_test, Fs) * exp(1j * 2 * pi * fd_test * t_slow);
            end
            mix_test = tx .* conj(rx_template_test);
            
            joint_energy = abs(sum(residual_mix .* conj(mix_test)));
            if joint_energy > best_joint_energy
                best_joint_energy = joint_energy;
                best_r = r_test;
                best_v = v_test;
            end
        end
    end
    
    rough_detected_ranges(i) = best_r;
    rough_detected_velocities(i) = best_v;
    
    fprintf(' Rough Range %.2f m  | Velocity %.2f m/s \n', ...
         rough_detected_ranges(i), rough_detected_velocities(i) );
    
    tau_est = 2 * rough_detected_ranges(i) / c;
    fd_est = 2 * rough_detected_velocities(i) / lambda;
    rx_template_remove = zeros(size(tx));
    for m_ch = 1:N_chirps
        t_slow = (m_ch - 1) * Tc;
        idx_s = (m_ch-1) * N_samples + 1;
        idx_e = m_ch * N_samples;
        rx_template_remove(idx_s:idx_e) = delayseq(tx_single, tau_est, Fs) * exp(1j * 2 * pi * fd_est * t_slow);
    end
    target_mix_est = tx .* conj(rx_template_remove);
    residual_mix = residual_mix - target_mix_est;
end

[rough_detected_ranges, sort_idx] = sort(rough_detected_ranges);
rough_detected_velocities = rough_detected_velocities(sort_idx);

%% 5. IFF 
N_total = length(mix);
t_total = (0:N_total-1)' / Fs;
detected_ranges = zeros(1, num_targets);
detected_velocities = zeros(1, num_targets);
sigma = 0.005;
K = 5;

for target_idx = 1:num_targets
    rough_center = rough_detected_ranges(target_idx);
    rough_v_center = rough_detected_velocities(target_idx);
    
    isolated_mix_target = mix;
    for other_idx = 1:num_targets
        if other_idx ~= target_idx
            tau_other = 2 * rough_detected_ranges(other_idx) / c;
            fd_other = 2 * rough_detected_velocities(other_idx) / lambda;
            
           
            rx_template_other = zeros(size(tx));
            for m_ch = 1:N_chirps
                t_slow = (m_ch - 1) * Tc;
                idx_s = (m_ch-1) * N_samples + 1;
                idx_e = m_ch * N_samples;
                rx_template_other(idx_s:idx_e) = delayseq(tx_single, tau_other, Fs) * exp(1j * 2 * pi * fd_other * t_slow);
            end
            isolated_mix_target = isolated_mix_target - (tx .* conj(rx_template_other));
        end
    end
    
    zeta_local = angle(isolated_mix_target(2:end) .* conj(isolated_mix_target(1:end-1))) / (2*pi) * Fs;
    t_zeta = t_total(1:end-1);
    
    zeta_IFF = zeta_local;
    t_abs = t_zeta;
    a_t = S * mod(t_abs, Tc);  
    
    range_grid = linspace(rough_center - 0.3, rough_center + 0.3, 1000);
    LIFF_fine = zeros(size(range_grid));
    fd_fixed = 2 * rough_v_center / lambda; 
    
    for ir = 1:length(range_grid)
        R_trial = range_grid(ir);
        tau_trial = 2 * R_trial / c;
        
        t_delayed_abs = t_abs - tau_trial;
        a_rx = S * mod(t_delayed_abs, Tc);
        valid_mask = t_abs >= tau_trial;
        a_rx(~valid_mask) = 0;
        
        g_model = a_t - a_rx + fd_fixed;
        g_norm = mod(g_model/Fs + 0.5, 1) - 0.5;
        
        LIFF_fine(ir) = iff_likelihood(zeta_IFF, g_norm, sigma, K);
    end
    
    [~, fine_idx] = max(LIFF_fine);
    detected_ranges(target_idx) = range_grid(fine_idx);
    
    velocity_grid = linspace(rough_v_center - 0.5, rough_v_center + 0.5, 500);
    LIFF_v_fine = zeros(size(velocity_grid));
    
    tau_locked = 2 * detected_ranges(target_idx) / c;
    t_delayed_abs_locked = t_abs - tau_locked;
    a_rx_locked = S * mod(t_delayed_abs_locked, Tc);
    valid_mask_locked = t_abs >= tau_locked;
    a_rx_locked(~valid_mask_locked) = 0;
    
    for iv = 1:length(velocity_grid)
        v_trial = velocity_grid(iv);
        fd_trial = 2 * v_trial / lambda; 
        
        g_model_v = a_t - a_rx_locked + fd_trial;
        g_norm_v = mod(g_model_v/Fs + 0.5, 1) - 0.5;
        
        LIFF_v_fine(iv) = iff_likelihood(zeta_IFF, g_norm_v, sigma, K);
    end
    
    [~, fine_v_idx] = max(LIFF_v_fine);
    detected_velocities(target_idx) = velocity_grid(fine_v_idx);
end

for i = 1:num_targets
    fprintf('Target %d Results:\n', i);
    fprintf('  - Estimated Range   : %.2f m \n', detected_ranges(i) );
    fprintf('  - Estimated Velocity: %.2f m/s \n', detected_velocities(i));
end

%% 6. IFF LIKELIHOOD FUNCTION
function L = iff_likelihood(zeta, g_norm, sigma, K)
    Fs = 60e6;
    zeta_norm = zeta / Fs;
    zeta_wrapped = mod(zeta_norm + 0.5, 1) - 0.5;
    diff = zeta_wrapped - g_norm;
    prob = zeros(size(diff));
    for k = -K:K
        prob = prob + exp(-(diff - k).^2 / (2 * sigma^2));
    end
    L = sum(log(prob + eps));
end