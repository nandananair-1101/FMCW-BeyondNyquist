clear;
clc;
close all;
tic;
fprintf('Starting Fast Monte Carlo Simulation with Blind Data-Driven Sigma Scaling (Parallelized)...\n');
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
fprintf('System Nyquist Unambiguous Range: %.1f m\n', R_nyquist);
%% 2. MULTI-TARGET & MONTE CARLO CONFIG
target_Rs = [65.0, 130.0, 240.0, 377.0, 422.0];        
target_vs = [-5.0, -10.0, -15.0, 5.0, 15.0];        
num_targets = length(target_Rs);
snr_vec = -50:5:20;          
num_monte_carlo = 5;         
rmse_pure_fft_range_agg = zeros(length(snr_vec), 1);
rmse_iff_range_agg      = zeros(length(snr_vec), 1);
rmse_pure_fft_vel_agg   = zeros(length(snr_vec), 1);
rmse_iff_vel_agg        = zeros(length(snr_vec), 1);
cdf_err_fft_r_10db = [];
cdf_err_iff_r_10db = [];
total_time_fft = 0;
total_time_iff = 0;
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
rx_clean = zeros(size(tx));
for m = 1:N_chirps
    t_slow = (m - 1) * Tc;
    rx_chirp_m = zeros(size(tx_single));
    for target_idx = 1:num_targets
        fd_true_m = 2 * target_vs(target_idx) / lambda;
        rx_single_target = delayseq(tx_single, tau_true(target_idx), Fs) * exp(1j * 2 * pi * fd_true_m * t_slow);
        rx_chirp_m = rx_chirp_m + rx_single_target;
    end
    rx_clean((m-1)*N_samples + 1 : m*N_samples) = rx_chirp_m;
end
%% 4. PRE-CALCULATE AXES & CONSTANTS
n_fft_range = 4096;
n_fft_doppler = 128;
max_search_range = 1000.0;
max_k = floor(max_search_range / R_nyquist); 
range_win = hann(N_samples);
doppler_win = hann(N_chirps)';
fft_freq_axis = (0 : n_fft_range-1)' * (Fs / n_fft_range);
fft_range_axis = fft_freq_axis * c / (2 * S);
PRF = 1 / Tc;
doppler_freqs = (-n_fft_doppler/2 : n_fft_doppler/2 - 1) * (PRF / n_fft_doppler);
rdm_velocity_axis = doppler_freqs * (lambda / 2);
rough_velocity_axis = linspace(-50, 50, 21); 
N_total = length(tx);
t_total = (0:N_total-1)' / Fs;
%% 5. MONTE CARLO OVER SNR 
for snr_idx = 1:length(snr_vec)
    current_snr = snr_vec(snr_idx);
    fprintf('\n-----------------------------------------\n');
    fprintf('Running Monte Carlo for SNR = %d dB...\n', current_snr);
    fprintf('-----------------------------------------\n');
    
    err_fft_r_mc = zeros(num_monte_carlo, 1);
    err_iff_r_mc = zeros(num_monte_carlo, 1);
    err_fft_v_mc = zeros(num_monte_carlo, 1);
    err_iff_v_mc = zeros(num_monte_carlo, 1);
    
    time_fft_mc = zeros(1, num_monte_carlo);
    time_iff_mc = zeros(1, num_monte_carlo);
    
    local_err_fft_r = cell(num_monte_carlo, 1);
    local_err_iff_r = cell(num_monte_carlo, 1);
    
    parfor mc = 1:num_monte_carlo
        local_mix = awgn(tx .* conj(rx_clean), current_snr, 'measured');
        
        %% --- METHOD 1: PURE NORMAL FFT ---
        t1 = tic;
        
        Mix_matrix = reshape(local_mix, N_samples, N_chirps);
        Mix_windowed = Mix_matrix .* range_win .* doppler_win;
        RDM_range = fft(Mix_windowed, n_fft_range, 1);
        RDM_pure = abs(fftshift(fft(RDM_range, n_fft_doppler, 2), 2));
        
        raw_fft_ranges = zeros(1, num_targets);
        raw_fft_vels   = zeros(1, num_targets);
        temp_RDM = RDM_pure;
        
        for i = 1:num_targets
            [~, max_lin_idx] = max(temp_RDM(:));
            [r_bin, c_bin] = ind2sub(size(temp_RDM), max_lin_idx);
            raw_fft_ranges(i) = fft_range_axis(r_bin); 
            raw_fft_vels(i)   = -rdm_velocity_axis(c_bin);
            temp_RDM(max(1, r_bin-5):min(n_fft_range, r_bin+5), :) = 0;
        end
        
        pure_fft_detected_ranges = zeros(1, num_targets);
        pure_fft_detected_vels   = zeros(1, num_targets);
        temp_pool_ranges = raw_fft_ranges;
        temp_pool_vels = raw_fft_vels;
        
        for t_idx = 1:num_targets
            [~, best_match_idx] = min(abs(temp_pool_ranges - target_Rs(t_idx)));
            pure_fft_detected_ranges(t_idx) = temp_pool_ranges(best_match_idx);
            pure_fft_detected_vels(t_idx)   = temp_pool_vels(best_match_idx);
            temp_pool_ranges(best_match_idx) = inf; 
        end
        
        time_fft_mc(mc) = toc(t1);
        
        %% --- METHOD 2: FFT + IFF ---
        t2 = tic;
        
        rough_detected_ranges = zeros(1, num_targets);
        rough_detected_velocities = zeros(1, num_targets);
        residual_mix = local_mix;
        
        for i = 1:num_targets
            Mix_matrix_r = reshape(residual_mix, N_samples, N_chirps);
            Mix_windowed_r = Mix_matrix_r .* range_win .* doppler_win;
            RDM_range_r = fft(Mix_windowed_r, n_fft_range, 1);
            RDM_r = abs(fftshift(fft(RDM_range_r, n_fft_doppler, 2), 2));
            
            [~, max_lin_idx_r] = max(RDM_r(:));
            [r_bin_r, c_bin_r] = ind2sub(size(RDM_r), max_lin_idx_r);
            
            R_alias = fft_range_axis(r_bin_r);
            candidate_ks = 0:max_k;
            candidate_ranges = R_alias + candidate_ks * R_nyquist;
            candidate_ranges = candidate_ranges(candidate_ranges <= max_search_range);
            
            best_joint_energy = -inf;
            best_r = R_alias;
            best_v = rdm_velocity_axis(c_bin_r); 
            
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
                    
                    joint_energy = abs(sum(residual_mix .* conj(mix_test))) / norm(mix_test);
                    if joint_energy > best_joint_energy
                        best_joint_energy = joint_energy;
                        best_r = r_test;
                        best_v = v_test;
                    end
                end
            end
            
            rough_detected_ranges(i) = best_r;
            rough_detected_velocities(i) = best_v;
            
            tau_est = 2 * rough_detected_ranges(i) / c;
            fd_est = 2 * rough_detected_velocities(i) / lambda;
            rx_template_remove = zeros(size(tx));
            for m_ch = 1:N_chirps
                t_slow = (m_ch - 1) * Tc;
                idx_s = (m_ch-1) * N_samples + 1;
                idx_e = m_ch * N_samples;
                rx_template_remove(idx_s:idx_e) = delayseq(tx_single, tau_est, Fs) * exp(1j * 2 * pi * fd_est * t_slow);
            end
            residual_mix = residual_mix - (tx .* conj(rx_template_remove));
        end
        [rough_detected_ranges, sort_idx] = sort(rough_detected_ranges);
        rough_detected_velocities = rough_detected_velocities(sort_idx);
        
        detected_ranges = zeros(1, num_targets);
        detected_vels   = zeros(1, num_targets); 
        
        % --- DATA-DRIVEN BLIND SNR & SIGMA ESTIMATION ---
        noise_floor_est = median(abs(RDM_pure(:))) / 0.6745; 
        peak_signal_val = max(RDM_pure(:));
        est_linear_snr = max(1, (peak_signal_val / noise_floor_est)^2);
        
        sigma = max(1e-5, 0.01 / sqrt(est_linear_snr));
        K = 5;
        
        for target_idx = 1:num_targets
            rough_center = rough_detected_ranges(target_idx);
            rough_v_center = rough_detected_velocities(target_idx);
            
            isolated_mix_target = local_mix;
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
            t_abs = t_total(1:end-1);
            a_t = S * mod(t_abs, Tc);  
            
            % Fine Range Optimization using fminbnd (Sub-millimeter precision)
            options = optimset('TolX', 1e-8, 'Display', 'off');
            fd_fixed = 2 * rough_v_center / lambda; 
            
            obj_r = @(R) -iff_likelihood(zeta_local, ...
                mod((a_t - (S * mod(t_abs - (2*R/c), Tc)) + fd_fixed)/Fs + 0.5, 1) - 0.5, ...
                sigma, K);
                
            best_r_refined = fminbnd(obj_r, rough_center - 0.2, rough_center + 0.2, options);
            detected_ranges(target_idx) = best_r_refined;
           
            
            tau_locked = 2 * detected_ranges(target_idx) / c;
            t_delayed_abs_locked = t_abs - tau_locked;
            a_rx_locked = S * mod(t_delayed_abs_locked, Tc);
            valid_mask_locked = t_abs >= tau_locked;
            a_rx_locked(~valid_mask_locked) = 0;
            
            obj_v = @(v) -iff_likelihood(zeta_local, ...
                mod((a_t - a_rx_locked + (2*v/lambda))/Fs + 0.5, 1) - 0.5, ...
                sigma, K);
                
            best_v_refined = fminbnd(obj_v, rough_v_center - 0.2, rough_v_center + 0.2, options);
            detected_vels(target_idx) = best_v_refined;
        end
        
        time_iff_mc(mc) = toc(t2);
        
        %% OUTPUTS
        if mc == num_monte_carlo
            fprintf('  -> [SNR %d dB] True Ranges:        [%.1f, %.1f, %.1f, %.1f, %.1f] m\n', current_snr, target_Rs(1), target_Rs(2), target_Rs(3), target_Rs(4), target_Rs(5));
            fprintf('  -> [SNR %d dB] True Velocities:    [%.1f, %.1f, %.1f, %.1f, %.1f] m/s\n', current_snr, target_vs(1), target_vs(2), target_vs(3), target_vs(4), target_vs(5));
            fprintf('  -> [SNR %d dB] Pure FFT Ranges:    [%.1f, %.1f, %.1f, %.1f, %.1f] m\n', current_snr, pure_fft_detected_ranges(1), pure_fft_detected_ranges(2), pure_fft_detected_ranges(3), pure_fft_detected_ranges(4), pure_fft_detected_ranges(5));
            fprintf('  -> [SNR %d dB] Pure FFT Vels:      [%.1f, %.1f, %.1f, %.1f, %.1f] m/s\n', current_snr, pure_fft_detected_vels(1), pure_fft_detected_vels(2), pure_fft_detected_vels(3), pure_fft_detected_vels(4), pure_fft_detected_vels(5));
            fprintf('  -> [SNR %d dB] Proposed IFF Ranges:[%.1f, %.1f, %.1f, %.1f, %.1f] m\n', current_snr, detected_ranges(1), detected_ranges(2), detected_ranges(3), detected_ranges(4), detected_ranges(5));
            fprintf('  -> [SNR %d dB] Proposed IFF Vels:    [%.2f, %.2f, %.2f, %.2f, %.2f] m/s\n', current_snr, detected_vels(1), detected_vels(2), detected_vels(3), detected_vels(4), detected_vels(5));
        end
        
        %% ERROR CALC
        err_fft_r_target = (pure_fft_detected_ranges - target_Rs).^2;
        err_iff_r_target = (detected_ranges - target_Rs).^2;
        
        err_fft_v_target = (pure_fft_detected_vels - target_vs).^2;
        err_iff_v_target = (detected_vels - target_vs).^2;
        
        err_fft_r_mc(mc) = mean(err_fft_r_target);
        err_iff_r_mc(mc) = mean(err_iff_r_target);
        err_fft_v_mc(mc) = mean(err_fft_v_target);
        err_iff_v_mc(mc) = mean(err_iff_v_target);
        
        local_err_fft_r{mc} = abs(pure_fft_detected_ranges - target_Rs);
        local_err_iff_r{mc} = abs(detected_ranges - target_Rs);
    end
    
    rmse_pure_fft_range_agg(snr_idx) = sqrt(mean(err_fft_r_mc));
    rmse_iff_range_agg(snr_idx)      = sqrt(mean(err_iff_r_mc));
    rmse_pure_fft_vel_agg(snr_idx)   = sqrt(mean(err_fft_v_mc));
    rmse_iff_vel_agg(snr_idx)        = sqrt(mean(err_iff_v_mc));
    
    snr_err_fft_r = horzcat(local_err_fft_r{:});
    snr_err_iff_r = horzcat(local_err_iff_r{:});
    
    % Isolate errors strictly for SNR = 10 dB
    if current_snr == 10
        cdf_err_fft_r_10db = snr_err_fft_r;
        cdf_err_iff_r_10db = snr_err_iff_r;
    end
    
    total_time_fft = total_time_fft + sum(time_fft_mc);
    total_time_iff = total_time_iff + sum(time_iff_mc);
end
elapsed_time = toc;
%% 6. RMSE PLOT 
figure('Color', 'black', 'Position', [100, 100, 1300, 550]);
subplot(1,2,1);
plot(snr_vec, rmse_pure_fft_range_agg, '-o', 'LineWidth', 2, 'MarkerFaceColor', 'auto'); hold on;
plot(snr_vec, rmse_iff_range_agg, '-s', 'LineWidth', 2, 'MarkerFaceColor', 'auto');
grid on; set(gca, 'YScale', 'log');
xlabel('SNR (dB)', 'Color', 'white'); ylabel('Aggregate Range RMSE (m) [Log Scale]', 'Color', 'white');
title('Aggregate Range RMSE (5 Targets)', 'Color', 'white');
legend('Pure Normal FFT', 'Proposed IFF Method', 'Location', 'northeast');
set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w');
subplot(1,2,2);
plot(snr_vec, rmse_pure_fft_vel_agg, '-o', 'LineWidth', 2, 'MarkerFaceColor', 'auto'); hold on;
plot(snr_vec, rmse_iff_vel_agg, '-s', 'LineWidth', 2, 'MarkerFaceColor', 'auto');
grid on; set(gca, 'YScale', 'log');
xlabel('SNR (dB)', 'Color', 'white'); ylabel('Aggregate Velocity RMSE (m/s) [Log Scale]', 'Color', 'white');
title('Aggregate Velocity RMSE (5 Targets)', 'Color', 'white');
legend('Pure Normal FFT', 'Proposed IFF Method', 'Location', 'northeast');
set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w');
set(gcf, 'InvertHardcopy', 'off');
%% 7. CDF PLOT AT FIXED SNR = 10 dB
figure('Color', 'black', 'Position', [150, 150, 800, 500]);
[x_fft, ~] = sort(cdf_err_fft_r_10db(:));
cdf_fft = (1:length(x_fft))' / length(x_fft);
[x_iff, ~] = sort(cdf_err_iff_r_10db(:));
cdf_iff = (1:length(x_iff))' / length(x_iff);
plot(x_fft, cdf_fft, 'm--', 'LineWidth', 2); hold on;
plot(x_iff, cdf_iff, 'c-', 'LineWidth', 2);
grid on;
xlabel('Absolute Range Error (m)', 'Color', 'white');
ylabel('Cumulative Probability (CDF)', 'Color', 'white');
title('CDF of Range Errors at SNR = 10 dB (Pure FFT vs. Proposed IFF)', 'Color', 'white');
legend('Pure Normal FFT', 'Proposed IFF Method', 'Location', 'southeast');
set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w');
set(gcf, 'InvertHardcopy', 'off');
%% 8. IFF LIKELIHOOD FUNCTION
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
