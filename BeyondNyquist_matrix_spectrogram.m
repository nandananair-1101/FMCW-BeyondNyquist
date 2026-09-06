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
v_max = lambda / (4 * Tc);

N_chirps = 64; 


R_nyquist = (c * Fs/2) / (2 * S);

%% 2. TARGET - BEYOND NYQUIST 

target_R = 280.0;
target_v = 0.0;

tau_true = 2*target_R/c;
fd_true = 2*target_v/lambda;
f_beat_true = S * tau_true;

fprintf('=== TARGET ===\n');
fprintf('True Range       : %.1f m (BEYOND Nyquist!)\n', target_R);
fprintf('True Beat Freq   : %.3f MHz\n', f_beat_true/1e6);
fprintf('True Doppler     : %.2f Hz\n', fd_true);
fprintf('Nyquist Freq     : %.1f MHz\n', Fs/2/1e6);
fprintf('\n');



%% 3. GENERATE  SIGNAL

waveform = phased.FMCWWaveform( ...
    'SweepTime', Tc, ...
    'SweepBandwidth', B, ...
    'SampleRate', Fs, ...
    'SweepDirection', 'Up', ...
    'NumSweeps', N_chirps);

tx = waveform();
t = (0:length(tx)-1)' / Fs;

% Target
rx = delayseq(tx, tau_true, Fs);
rx = rx .* exp(1j*2*pi*fd_true*t);
mix = tx .* conj(rx);
Mix = reshape(mix, N_samples, N_chirps);

%% 4. DIAGNOSTIC PLOTS - FIGURE 1

Fs_plot = 300e6;
waveform_plot = phased.FMCWWaveform( ...
    'SweepTime', Tc, ...
    'SweepBandwidth', B, ...
    'SampleRate', Fs_plot, ...
    'SweepDirection', 'Up', ...
    'NumSweeps', 2);

tx_p = waveform_plot();
t_plot = (0:length(tx_p)-1)' / Fs_plot;

tau_compare = 2*45/c;
fd_compare = 0;

rx_p_target = delayseq(tx_p, tau_true, Fs_plot) .* exp(1j*2*pi*fd_true*t_plot);
rx_p_compare = delayseq(tx_p, tau_compare, Fs_plot) .* exp(1j*2*pi*fd_compare*t_plot);

mix_p_target = tx_p .* conj(rx_p_target);
mix_p_compare = tx_p .* conj(rx_p_compare);

figure('Name','FMCW CBF Aliasing Proof (45m vs 300m)','Position',[50 50 1100 750]);

% Tx vs Rx1 (45m)
subplot(3,2,1);
plot(t_plot(1:min(500,end))*1e6, real(tx_p(1:min(500,end))), 'b', 'LineWidth',1);
hold on;
plot(t_plot(1:min(500,end))*1e6, real(rx_p_compare(1:min(500,end))), 'r--', 'LineWidth',1);
xlim([0 0.8]);
title('Tx vs Rx (45 m - Within Nyquist)');
xlabel('Fast-Time (\mus)');
legend('Tx','Rx (45m)','Location','Northwest');
grid on;

% Tx vs Rx2 (300m)
subplot(3,2,2);
n_plot_300 = min(1500, length(tx_p));
plot(t_plot(1:n_plot_300)*1e6, real(tx_p(1:n_plot_300)), 'b', 'LineWidth',1);
hold on;
plot(t_plot(1:n_plot_300)*1e6, real(rx_p_target(1:n_plot_300)), 'm--', 'LineWidth',1);
xlim([0 2.5]);
title('Tx vs Rx (300 m - BEYOND Nyquist)');
xlabel('Fast-Time (\mus)');
legend('Tx','Rx (300m)','Location','Northwest');
grid on;

% IF Beat Output Overlay
subplot(3,2,3);
n_plot = min(500, N_samples);
plot((0:n_plot-1)/Fs*1e6, real(Mix(1:n_plot,1)), 'k', 'LineWidth',1);
hold on;
plot(t_plot(1:min(500,end))*1e6, real(mix_p_compare(1:min(500,end))), 'r--', 'LineWidth',1);
xlim([0 3]);
title('IF Beat Output Overlay');
xlabel('Fast-Time (\mus)');
legend('IF 300m','IF 45m (compare)');
grid on;

% Tx Spectrogram
subplot(3,2,4);
spectrogram(tx_p, 256, 250, 256, Fs_plot, 'yaxis');
title('Tx Spectrogram');
colorbar off;

% IF Mixed Output - 45m
subplot(3,2,5);
spectrogram(mix_p_compare, 256, 250, 256, Fs_plot, 'yaxis');
title('IF Mixed Output - 45m (Within Nyquist)');
colorbar off;

% IF Mixed Output - 300m
subplot(3,2,6);
spectrogram(mix_p_target, 256, 250, 256, Fs_plot, 'yaxis');
title('IF Mixed Output - 300m (BEYOND Nyquist - Aliased!)');
colorbar off;

sgtitle('FMCW Aliasing Proof: 45m vs 300m');



%% 5. CBF PROCESSING - SHOWS ALIASING
mix_chirp = Mix(:,1);

N_fft = 2^nextpow2(N_samples);
freq_axis = (-N_fft/2:N_fft/2-1) * Fs / N_fft / 1e6;
mix_fft = fftshift(fft(mix_chirp, N_fft));

search_region = abs(freq_axis) > 0.5;
fft_mag = abs(mix_fft);
fft_mag(~search_region) = 0;

[~, peak_idx] = max(fft_mag);
f_beat_cbf = freq_axis(peak_idx) * 1e6;
R_cbf = abs(f_beat_cbf) * c / (2 * S);



%% 6. EXTRACT INSTANTANEOUS FREQUENCY - PAPER EQ (41)

N_total = N_samples * N_chirps;
t_total = (0:N_total-1)' / Fs;

% Extract IF from the full signal
zeta = angle(mix(2:end) .* conj(mix(1:end-1))) / (2*pi) * Fs;
t_zeta = t_total(1:end-1);


trim = 5;
zeta_trim = zeta(trim:end-trim);
t_trim = t_zeta(trim:end-trim);

t_abs_IFF = t_trim;

%% 7. DOWNSAMPLE FOR IFF

dec = 10;
zeta_IFF = zeta_trim(1:dec:end);
t_IFF = t_trim(1:dec:end);

t_abs = t_IFF;
a_t = S * mod(t_abs, Tc);  

%% 8. IFF SEARCH 

range_grid = linspace(0, 350, 351);
f_grid = linspace(-8000, 8000, 31);

sigma = 0.005;
K = 5;

LIFF = zeros(length(range_grid), length(f_grid));


for ir = 1:length(range_grid)
    
    R_trial = range_grid(ir);
    tau_trial = 2*R_trial/c;
    
    for jf = 1:length(f_grid)
        fd_trial = f_grid(jf);
        
        t_delayed_abs = t_abs - tau_trial;
        a_rx = S * mod(t_delayed_abs, Tc);
                
        valid_mask = t_abs >= tau_trial;
        a_rx(~valid_mask) = 0;
        
        % Eq (38)
        g_model = a_t - a_rx + fd_trial;
        g_norm = mod(g_model/Fs + 0.5, 1) - 0.5;
        
        LIFF(ir,jf) = iff_likelihood(zeta_IFF, g_norm, sigma, K);
        
    end    
    
end


%% 9. FIND GLOBAL MAXIMUM

[~, idx] = max(LIFF(:));
[row, col] = ind2sub(size(LIFF), idx);
R_iff = range_grid(row);
fd_iff = f_grid(col);

fprintf('\n=== IFF RESULT ===\n');
fprintf('IFF Range        : %.2f m\n', R_iff);
fprintf('IFF Doppler      : %.2f Hz\n', fd_iff);
fprintf('True Range       : %.2f m\n', target_R);
fprintf('True Doppler     : %.2f Hz\n', fd_true);
fprintf('Range Error      : %.2f m\n', abs(R_iff - target_R));
fprintf('Doppler Error    : %.2f Hz\n', abs(fd_iff - fd_true));

R_final = R_iff;
fd_final = fd_iff;

%% 11. RESULTS SUMMARY

fprintf('====FINAL RESULTS====\n');
fprintf('True Range       : %.2f m\n', target_R);
fprintf('Nyquist Range    : %.2f m\n', R_nyquist);
fprintf('CBF Estimate     : %.2f m \n', R_cbf);
fprintf('IFF Estimate     : %.2f m\n', R_final);
fprintf('IFF Doppler      : %.2f Hz\n', fd_final);
fprintf('\n');

%% 12. IFF RESULTS FIGURE 

figure('Name','Up-Chirp IFF Beyond Nyquist','Position',[50 50 1400 700]);

% Subplot 1: IF Signal 
subplot(2,3,1);
n_plot = min(400, N_samples);
plot(t(1:n_plot)*1e6, real(mix(1:n_plot)), 'b-', 'LineWidth',1.2);
xlabel('Time (\mus)');
ylabel('Amplitude');
title('Dechirped IF Signal (First Chirp)');
grid on;

% Subplot 2: CBF Spectrum
subplot(2,3,2);
plot(freq_axis, 20*log10(abs(mix_fft)/max(abs(mix_fft))+eps), 'b-', 'LineWidth',1.5);
hold on;
xline(f_beat_true/1e6, 'g--', 'True f_b', 'LineWidth',2);
xline(f_beat_cbf/1e6, 'r--', 'Aliased f_b', 'LineWidth',2);
xlabel('Frequency (MHz)');
ylabel('Magnitude (dB)');
title('CBF Spectrum: Aliased!');
xlim([-40 40]);
ylim([-60 5]);
legend('FFT', 'True f_b', 'Aliased f_b', 'Location','best');
grid on;

% Subplot 3: Range Comparison
subplot(2,3,3);
bar_data = [target_R, R_cbf, R_final];
bar_colors = {[0 0.7 0], [0.9 0.2 0.2], [0 0.2 0.8]};
for i = 1:3
    bar(i, bar_data(i), 'FaceColor', bar_colors{i}, 'EdgeColor', 'k', 'LineWidth', 1.5);
    hold on;
end
set(gca, 'XTick',1:3, 'XTickLabel',{'True','CBF','IFF'});
ylabel('Range (m)');
title('Range Comparison');
ylim([0 350]);
grid on;
for i = 1:3
    text(i, bar_data(i)+15, sprintf('%.1f m',bar_data(i)), ...
        'HorizontalAlignment','center', 'FontWeight','bold');
end

% Subplot 4: IFF Likelihood Surface
subplot(2,3,4);
LIFF_dB = 10*log10(exp(LIFF - max(LIFF(:))) + eps);
imagesc(f_grid, range_grid, LIFF_dB);
axis xy;
colormap('jet');
colorbar;
caxis([-30 0]);
hold on;
plot(fd_iff, R_iff, 'wx', 'MarkerSize',14, 'LineWidth',3);
plot(fd_true, target_R, 'ro', 'MarkerSize',9, 'LineWidth',2);
xlabel('Doppler (Hz)');
ylabel('Range (m)');
title('IFF Likelihood Surface');
legend('IFF Estimate', 'True Target', 'Location','best');
grid on;

% Subplot 5: IFF Range Profile
subplot(2,3,5);
[~, col_idx] = min(abs(f_grid - fd_iff));
profile = LIFF(:, col_idx);
profile_dB = 10*log10(exp(profile - max(profile)) + eps);
plot(range_grid, profile_dB, 'b-', 'LineWidth',2);
hold on;
xline(target_R, 'r--', 'True', 'LineWidth',2);
xline(R_final, 'g--', 'IFF', 'LineWidth',2);
xlabel('Range (m)');
ylabel('Likelihood (dB)');
title('IFF Range Profile');
legend('IFF', 'True', 'Estimate', 'Location','best');
grid on;

% Subplot 6: Measured IF (Full Burst)
subplot(2,3,6);
n_plot = min(3000, length(t_trim));
plot(t_trim(1:n_plot)*1e6, zeta_trim(1:n_plot)/1e6, 'b-', 'LineWidth',1);
xlabel('Time (\mus)');
ylabel('IF (MHz)');
title('Measured IF (Full Burst)');
ylim([-35 35]);
grid on;

sgtitle(sprintf('Up-Chirp IFF: Target at %.0fm (Beyond %.0fm Nyquist)', target_R, R_nyquist), ...
        'FontSize',14, 'FontWeight','bold');


%% 13. IFF LIKELIHOOD FUNCTION - PAPER EQ (45)

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
%% Matrix_Analysis
target_Rs = [600.0, 300.0, 170.0]; 
tau_true = 2 * target_Rs / c; 
fig_handle = figure('Name', 'Multi-Target FMCW Analysis', 'Position', [50, 50, 1400, 900], 'Color', 'black');

for k = 1:length(target_Rs)
    
    waveform_multi = phased.FMCWWaveform('SweepTime', Tc, 'SweepBandwidth', B, 'SampleRate', Fs, 'SweepDirection', 'Up', 'NumSweeps', 2);
    full_chirp = waveform_multi();
    delayed_chirp = delayseq(full_chirp, tau_true(k), Fs);
    mix_chirp = full_chirp .* conj(delayed_chirp);
    
    N_sweep = round(Tc * Fs);
    sweep2_mix = mix_chirp(N_sweep + 1 : 2 * N_sweep);
    N_samples = length(sweep2_mix);
    
    window_len = 64;
    local_freqs = zeros(1, N_samples - window_len);
    for i = 1:(N_samples - window_len)
        local_part = sweep2_mix(i : i + window_len - 1);
        [~, loc] = max(abs(fft(local_part, 512)));
        local_freqs(i) = (loc - 1) * (Fs / 512);
    end
    [~, max_jump_idx] = max(abs(diff(local_freqs)));
    transition_idx = max_jump_idx + round(window_len / 2);
    transition_idx = max(min(transition_idx, N_samples - 50), 50);
    
    seg_noncbf = sweep2_mix(1:transition_idx);         
    seg_cbf    = sweep2_mix(transition_idx + 1:end);     
    
    N_fft = 2^nextpow2(N_samples);
    freq_axis = (-N_fft/2:N_fft/2-1) * Fs / N_fft;
    search_region = abs(freq_axis) > 0.5e6;
    
    fft_cbf = fftshift(fft(seg_cbf, N_fft));
    mag_cbf = abs(fft_cbf);
    mag_cbf_search = mag_cbf; mag_cbf_search(~search_region) = 0;
    [~, p1_sol] = max(mag_cbf_search);
    f_1_measured = abs(freq_axis(p1_sol));
    
    fft_noncbf = fftshift(fft(seg_noncbf, N_fft));
    mag_noncbf = abs(fft_noncbf); 
    mag_noncbf_search = mag_noncbf; mag_noncbf_search(~search_region) = 0;
    [~, p2_sol] = max(mag_noncbf_search); 
    f_2_measured = abs(freq_axis(p2_sol));
    
    f_3 = 1 / (length(seg_noncbf) / Fs);
    f_4 = 1 / (length(seg_cbf) / Fs);
    
    Y = [f_1_measured;
         f_2_measured + B;
         1/f_3; 
         Tc - 1/f_4];
    H = [S, -Fs, 0; 
         S, 0, -Fs; 
         1, 0, 0;
         1, 0, 0];
    
    H([3,4],:) = H([3,4],:) * Fs; Y([3,4]) = Y([3,4]) * Fs;
    
    X_est = pinv(H) * Y;
    est_range = abs(X_est(1)) * c / 2;
    fprintf('Target %.1f m - Estimated: %.2f m | f1: %.2f MHz | f2: %.2f MHz| f3: %.2f MHz\n', target_Rs(k), est_range, f_1_measured/1e6, f_2_measured/1e6,f_3/1e6);

    sub_pos = 2*k - 1;
    if sub_pos <= 5
        figure(fig_handle);
        
        subplot(3, 2, sub_pos);
        plot(freq_axis / 1e6, mag_cbf, 'LineWidth', 1.2);
        title(sprintf('Target %.1f m: CBF (f_1 = %.2f MHz)', target_Rs(k), f_1_measured/1e6));
        xlabel('Frequency (MHz)'); ylabel('Magnitude'); grid on;
        
        subplot(3, 2, sub_pos + 1);
        plot(freq_axis / 1e6, mag_noncbf, 'r', 'LineWidth', 1.2);
        title(sprintf('Target %.1f m: Non-CBF (f_2 = %.2f MHz)', target_Rs(k), f_2_measured/1e6));
        xlabel('Frequency (MHz)'); ylabel('Magnitude'); grid on;
    end

  
end


%% multi target
target_Rs = [600.0, 300.0];       
tau_true = 2 * target_Rs / c;     


waveform_multi = phased.FMCWWaveform('SweepTime', Tc, 'SweepBandwidth', B, ...
    'SampleRate', Fs, 'SweepDirection', 'Up', 'NumSweeps', 2);
full_chirp = waveform_multi();

delayed_chirp_sum = zeros(size(full_chirp));
for m = 1:length(target_Rs)
    delayed_chirp_sum = delayed_chirp_sum + delayseq(full_chirp, tau_true(m), Fs);
end

mix_chirp = full_chirp .* conj(delayed_chirp_sum);


window_size = 256;
overlap_samples = 250;
nfft = 256;

[S_mix, f, t] = spectrogram(mix_chirp, window_size, overlap_samples, nfft, Fs, 'centered');

fprintf('--- Multi-Target Direct Range Estimation via Spectrogram Gap ---\n');

mean_spectrum = mean(abs(S_mix), 2);
[pks, locs] = findpeaks(mean_spectrum, 'MinPeakHeight', max(mean_spectrum)*0.2, 'NPeaks', 2);

estimated_ranges = zeros(size(target_Rs));

for m = 1:length(locs)
    target_freq_bin = locs(m);       
       
    row_profile = abs(S_mix(target_freq_bin, :));    
    baseline_val = median(row_profile);
    
    
    gap_indices = find(row_profile < baseline_val * 0.55 | row_profile > baseline_val * 1.6);
    
    if length(gap_indices) >= 2
        
        reset_time_idx = find(t >= 35e-6 & t <= 45e-6);
        target_gap_indices = intersect(gap_indices, reset_time_idx);
        
        if length(target_gap_indices) >= 2
            tau_measured = t(target_gap_indices(end)) - t(target_gap_indices(1));
        else
           
            tau_measured = t(gap_indices(end)) - t(gap_indices(1));
        end
    else
        tau_measured = 0;
    end
    
   
    estimated_ranges(m) = (c * tau_measured) / 2; 
    
    fprintf('Measured Tau: %.2f us | Estimated Range: %.1f m\n', ...
        tau_measured * 1e6, estimated_ranges(m));
end

% 5. Plot
Fs_plot = 300e6;
waveform_multi_plot = phased.FMCWWaveform('SweepTime', Tc, 'SweepBandwidth', B, 'SampleRate', Fs_plot, 'SweepDirection', 'Up', 'NumSweeps', 2);
full_chirp_plot = waveform_multi_plot();

delayed_chirp_sum_plot = zeros(size(full_chirp_plot));
for m = 1:length(target_Rs)
    delayed_chirp_sum_plot = delayed_chirp_sum_plot + delayseq(full_chirp_plot, tau_true(m), Fs_plot);
end
mix_chirp_plot = full_chirp_plot .* conj(delayed_chirp_sum_plot);

figure('Name','multi target','Position',[50 50 1100 750]);

% Tx Spectrogram
subplot(3,2,1);
spectrogram(full_chirp_plot, 256, 250, 256, Fs_plot, 'yaxis');
title('Tx Spectrogram');
colorbar off;

% Mix Spectrogram
subplot(3,2,2);
spectrogram(mix_chirp_plot, 256, 250, 256, Fs_plot, 'yaxis');
title('Mix Spectrogram');
colorbar off;