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
fprintf('System Nyquist Unambiguous Range: %.1f m\n', R_nyquist);

%% 2. MULTI-TARGET 
target_Rs = [400.0, 500.0];       
target_vs = [0.0, 0.0];
num_targets = length(target_Rs);


for i = 1:num_targets
    fprintf('Target %d -> True Range : %.1f m\n', i, target_Rs(i));
end

%% 3. SIGNAL GENERATION
tau_true = 2 * target_Rs / c;     
waveform_multi = phased.FMCWWaveform( ...
    'SweepTime', Tc, ...
    'SweepBandwidth', B, ...
    'SampleRate', Fs, ...
    'SweepDirection', 'Up', ...
    'NumSweeps', N_chirps);
tx = waveform_multi();
t = (0:length(tx)-1)' / Fs;

rx_total = zeros(size(tx));
for m = 1:num_targets
    fd_true_m = 2 * target_vs(m) / lambda;
    rx_target = delayseq(tx, tau_true(m), Fs) .* exp(1j * 2 * pi * fd_true_m * t);
    rx_total = rx_total + rx_target;
end
mix = tx .* conj(rx_total);
Mix = reshape(mix, N_samples, N_chirps);

%% 4. match filter
coarse_range_axis = linspace(10, 800, 1600); 
residual_mix = mix;
rough_detected_ranges = zeros(1, num_targets);

for i = 1:num_targets
    
    energy_profile = zeros(size(coarse_range_axis));
    
    
    for r_idx = 1:length(coarse_range_axis)
        r_test = coarse_range_axis(r_idx);
        tau_test = 2 * r_test / c;
        rx_template_test = delayseq(tx, tau_test, Fs);
        mix_test = tx .* conj(rx_template_test);
        
        energy_profile(r_idx) = abs(sum(residual_mix .* conj(mix_test)));
    end
    
    [~, peak_idx] = max(energy_profile);
    rough_detected_ranges(i) = coarse_range_axis(peak_idx);
    fprintf('   Iteration %d: Found Rough Region at %.2f m\n', i, rough_detected_ranges(i));
    
    
    tau_est = 2 * rough_detected_ranges(i) / c;
    rx_template = delayseq(tx, tau_est, Fs);
    target_mix_est = tx .* conj(rx_template);
    residual_mix = residual_mix - target_mix_est;
end

%% 5. IFF
N_total = length(mix);
t_total = (0:N_total-1)' / Fs;

detected_ranges = zeros(1, num_targets);
sigma = 0.005;
K = 5;
fd_fixed = 0.0;

for target_idx = 1:num_targets
    rough_center = rough_detected_ranges(target_idx);
    
    
    
    isolated_mix_target = mix;
    for other_idx = 1:num_targets
        if other_idx ~= target_idx
            tau_other = 2 * rough_detected_ranges(other_idx) / c;
            rx_template_other = delayseq(tx, tau_other, Fs);
            isolated_mix_target = isolated_mix_target - (tx .* conj(rx_template_other));
        end
    end
    
    zeta_local = angle(isolated_mix_target(2:end) .* conj(isolated_mix_target(1:end-1))) / (2*pi) * Fs;
    t_zeta = t_total(1:end-1);
    
    zeta_IFF = zeta_local;
    t_abs = t_zeta;
    a_t = S * mod(t_abs, Tc);  
    
   
    range_grid = linspace(max(10, rough_center - 15), min(800, rough_center + 15), 151);
    LIFF_fine = zeros(size(range_grid));
    
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
end


for i = 1:num_targets
    fprintf('Target %d:\n', i);
    fprintf('  - Estimated Range: %.2f m\n', detected_ranges(i));
    
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