clear; clc; close all;
%% 1. System Parameters
c = 3e8; 
fc = 77e9; 
lambda = c/fc;
delta_R = 0.1;
B = c / (2 * delta_R);
Fs = 60e6; % Fs>2fbeat max...According to qstn Rmax=100...((S*2*100)/c=25.532MHz..ans*2=51.064MHz..hence Fs=60MHz
N_samples = 2350; 
Tc = N_samples / Fs;
S = B / Tc; 
N_chirps = 64; 
v_max = lambda / (4 * Tc); % according to qstn Vmax=90Km/hr...after calculating we get 24.8687m/s=89.5273Km/hr
Fs_plot = 300e6; 
v_res = lambda / (2 * N_chirps * Tc);
win = hann(N_samples) * hann(N_chirps)';
%% 2. Figure 1: Diagnostics
target_R = 45.0; target_v = 15.0;

waveform = phased.FMCWWaveform( ...
    'SweepTime', Tc, ...
    'SweepBandwidth', B, ...
    'SampleRate', Fs, ...
    'SweepDirection', 'Up', ...
    'NumSweeps', N_chirps);

tx = waveform();              % Length = N_samples*N_chirps
tau = 2*target_R/c;
rx = delayseq(tx,tau,Fs);
fd = 2*target_v/lambda;
t = (0:length(tx)-1)'/Fs;
rx = rx .* exp(1j*2*pi*fd*t);
mix = tx .* conj(rx);
Mix = reshape(mix,N_samples,N_chirps);

waveform_plot = phased.FMCWWaveform( ...
    'SweepTime',Tc,...
    'SweepBandwidth',B,...
    'SampleRate',Fs_plot,...
    'SweepDirection','Up',...
    'NumSweeps',2);

tx_p = waveform_plot();
Fs_plot=waveform_plot.SampleRate;
rx_p = delayseq(tx_p,tau,Fs_plot);
t_plot = (0:length(tx_p)-1)'/Fs_plot;
rx_p = rx_p .* exp(1j*2*pi*fd*t_plot);


figure('Name', 'Figure 1', 'Position', [50, 50, 1100, 750]);
subplot(3,2,1); 
plot(t_plot*1e6, real(tx_p), 'b', t_plot*1e6, real(rx_p), 'r--'); 
xlim([0 ,0.8]); 
title('Tx vs Rx ');
xlabel('Fast-Time (\mu s)'); 
grid on;

subplot(3,2,2);
plot((0:N_samples-1)/Fs*1e6, real(Mix(:,1)), 'k'); 
xlim([0 3])
title('IF Beat Output');
xlabel('Fast-Time (\mu s)'); grid on;

subplot(3,2,3);
spectrogram(tx_p, 256, 250, 256, Fs_plot, 'yaxis'); 
title('Tx Spectrogram');

subplot(3,2,4); 
spectrogram(rx_p, 256, 250, 256, Fs_plot, 'yaxis');
title('Rx Spectrogram');

subplot(3,2,5);
spectrogram((Mix(:,1)), 256, 250, 256, Fs, 'yaxis'); 
title('IF Mixed Output');

%% 3. Figure 2: FFT Profiles

window_fast = hann(N_samples);
window_slow = hann(N_chirps);
window_2d   = window_fast*window_slow';

N_fft_padded = 4096;

R_max = Fs*c/(2*S);
range_axis = linspace(0,R_max,N_fft_padded);
vel_axis = linspace(v_max, -v_max, N_chirps);
rdm = fftshift(fft2(Mix.*window_2d,N_fft_padded,N_chirps),2);
rdm_db = 20*log10(abs(rdm)/max(abs(rdm(:))));

range_prof = max(rdm_db,[],2);
[peak_r_db,idx_r] = max(range_prof);
detected_range = range_axis(idx_r);
dop_prof = rdm_db(idx_r,:);
[peak_v_db,idx_v] = max(dop_prof);
detected_velocity = vel_axis(idx_v);

figure('Name','Figure 2: FFT Profiles','Position',[100 100 1200 420]);

subplot(1,3,1)
plot(range_axis,range_prof,'b','LineWidth',1.5)
hold on
plot(detected_range,peak_r_db,'ro','MarkerSize',8,'LineWidth',2)
text(detected_range+2,peak_r_db-3,sprintf('Peak = %.1f m',detected_range),'FontWeight','bold')
title('1D Range FFT')
xlabel('Range (m)')
ylabel('Normalized Power (dB)')
grid on
xlim([0 100])

subplot(1,3,2)
plot(vel_axis,dop_prof,'m','LineWidth',1.5)
hold on
plot(detected_velocity,peak_v_db,'ko','MarkerSize',8,'LineWidth',2)
text(detected_velocity+0.5, peak_v_db-3,sprintf('Peak = %.1f m/s',detected_velocity),'FontWeight','bold')
title('1D Doppler FFT')
xlabel('Velocity (m/s)')
ylabel('Normalized Power (dB)')
grid on

subplot(1,3,3)
imagesc(vel_axis,range_axis,rdm_db)
axis xy
colormap jet
colorbar
clim([-40 0])
hold on
plot(target_v,target_R,'wx', 'MarkerSize',12,'LineWidth',2)
title('2D Range-Doppler Map')
xlabel('Velocity (m/s)')
ylabel('Range (m)')
ylim([0 100])

%% 4. Figure 3: Multi-Target Resolution Matrix
window_fast = hann(N_samples);
window_slow = hann(N_chirps);
window_2d = window_fast * window_slow'; 

N_fft_padded = 4096;
freq_axis_pad = (0:N_fft_padded-1) * (Fs / N_fft_padded);
R_max = ( Fs* c) / (2 * S);
range_axis=linspace(0,R_max,N_fft_padded);
R_res = c / (2 * B); 
v_res = lambda / (2 * N_chirps * Tc);
fig3 = figure('Name', ' Multi-Target Resolution Matrix', 'Position', [50, 100, 1500, 600]);

for s = 1:5
    Mix_multi = zeros(N_samples, N_chirps);
    if s == 1, target_matrix = [10.0, 10.0; 40.0, -8.0]; scen_title = 'A: Distinct Profiles';
    elseif s == 2, target_matrix = [40.0, 5.0; 40.07, 5.0]; scen_title = 'B: Below Range Limit (7cm)';
    elseif s == 3, target_matrix = [40.0, 5.0; 40.20, 5.0]; scen_title = 'C: Above Range Limit (20cm)';
    elseif s == 4, target_matrix = [60.0, 12.0; 60.0, 12.4]; scen_title = 'D: Below Doppler Limit (0.4m/s)';
    else, target_matrix = [60.0, 12.0; 60.0, 14.5]; scen_title = 'E: Above Doppler Limit (2.5m/s)';
    end
    
   
    for i = 1:size(target_matrix,1)

    tau = 2*target_matrix(i,1)/c;
    fd  = 2*target_matrix(i,2)/lambda;
    t = (0:length(tx)-1)'/Fs;
    rx = delayseq(tx,tau,Fs);
    rx = rx .* exp(1i*2*pi*fd*t);
    mix = tx .* conj(rx);
    Mix_multi = Mix_multi + reshape(mix,N_samples,N_chirps);
    end
    
    rdm_multi = fftshift(fft2(Mix_multi .* window_2d, N_fft_padded, N_chirps), 2);
    rdm_multi_db = 20*log10(abs(rdm_multi)/max(abs(rdm_multi(:))));
    
    subplot(2, 3, s);
    imagesc(vel_axis, range_axis, rdm_multi_db);
    colormap('jet'); clim([-35 0]); axis xy; 
    hold on;
    
   
    plot(target_matrix(:,2), target_matrix(:,1), 'kx', 'MarkerSize', 10, 'LineWidth', 2);
    xlim([min(target_matrix(:,2))-3, max(target_matrix(:,2))+3]);
    ylim([min(target_matrix(:,1))-3, max(target_matrix(:,1))+3]);
    title(scen_title); xlabel('Velocity (m/s)'); ylabel('Range (m)');
    
end
%% RCS
%Reference Target

R_ref = 30;          
sigma_ref = 1;       

target1_R = 30;
target1_v = 10;
target1_sigma = 1;


target2_R = 50;
target2_v = -10;
target2_sigma = 0.8;

SNR_plot = -20:5:30;
fig4 = figure('Name','Range FFT vs SNR','Position',[100 100 1500 700]);
fig5 = figure('Name',' Doppler FFT vs SNR','Position',[100 100 1500 700]);
fig6 = figure('Name','Range Doppler Maps vs SNR','Position',[100 100 1400 900]);
plot_idx = 1;

for snr = SNR_plot

    Mix = zeros(N_samples,N_chirps);
    amp1 = sqrt(target1_sigma/sigma_ref)*(R_ref/target1_R)^2;
    amp2 = sqrt(target2_sigma/sigma_ref)*(R_ref/target2_R)^2;
    tau1 = 2*target1_R/c;
    fd1  = 2*target1_v/lambda;
    tau2 = 2*target2_R/c;
    fd2  = 2*target2_v/lambda;
    t = (0:length(tx)-1)'/Fs;

    rx1 = amp1*delayseq(tx,tau1,Fs);
    rx1 = rx1 .* exp(1j*2*pi*fd1*t);
    
    rx2 = amp2*delayseq(tx,tau2,Fs);
    rx2 = rx2 .* exp(1j*2*pi*fd2*t);

    Ps = mean(abs(rx1(:)).^2);
    Pn = Ps/(10^(snr/10));
    
    noise = sqrt(Pn/2)*(randn(size(rx1))+1j*randn(size(rx1)));
    rx = rx1 + rx2;
    rx_noisy = rx + noise;
    
    mix = tx .* conj(rx_noisy);
    Mix = reshape(mix,N_samples,N_chirps);

    window_fast = hann(N_samples);
    window_slow = hann(N_chirps);
    window_2d   = window_fast*window_slow';

    N_fft_padded = 4096;

    R_max = Fs*c/(2*S);
    range_axis = linspace(0,R_max,N_fft_padded);
    vel_axis = linspace(v_max, -v_max, N_chirps);
    rdm = fftshift(fft2(Mix.*window_2d,N_fft_padded,N_chirps),2);
    rdm_mag = abs(rdm);
    rdm_db = 20*log10(rdm_mag + eps);
    rdm_disp = rdm_db - max(rdm_db(:));

    range_prof = max(rdm_db,[],2);
    [peak_r_db,idx_r] = max(range_prof);
    detected_range = range_axis(idx_r);
    dop_prof = rdm_db(idx_r,:);
    [peak_v_db,idx_v] = max(dop_prof);
    detected_velocity = vel_axis(idx_v);
    

    figure(fig4)
    subplot(2,6,plot_idx)
    plot(range_axis,range_prof,'b','LineWidth',1.5)
    hold on
    plot(detected_range,peak_r_db,'ro','MarkerSize',6,'LineWidth',2)
    title(sprintf('Range FFT\nSNR = %d dB',snr))
    xlabel('Range (m)')
    ylabel('dB')
    grid on
    xlim([0 100])
    ylim([10 100])
    
    figure(fig5)
    subplot(2,6,plot_idx)
    plot(vel_axis,dop_prof,'m','LineWidth',1.5)
    hold on
    plot(detected_velocity,peak_v_db,'wo','MarkerSize',6,'LineWidth',2)
    title(sprintf('Doppler FFT\nSNR = %d dB',snr))
    xlabel('Velocity (m/s)')
    ylabel('dB')
    grid on
    
    figure(fig6)
    subplot(2,6,plot_idx)
    imagesc(vel_axis,range_axis,rdm_disp)
    axis xy
    colormap jet
    clim([-40 0])
    hold on
    plot(target1_v,target1_R,'wx','MarkerSize',10,'LineWidth',2)
    plot(target2_v,target2_R,'wo','MarkerSize',10,'LineWidth',2)
    title(sprintf('%d dB',snr))
    xlabel('Velocity (m/s)')
    ylabel('Range (m)')
    xlim([-15 15])
    ylim([20 70])

    plot_idx = plot_idx + 1;


end
%% same range diffferent doppler

SNR_list = [-5 15];
alpha_list = [0.1 0.25 0.5 1 2];

target1_R = 30;
target1_v = 10;
target1_sigma = 1;

target2_R = 30;      % SAME RANGE
target2_v = -10;     % DIFFERENT DOPPLER

for snr = SNR_list
    figure('Name',sprintf('Same Range Different Doppler - SNR = %d dB',snr),'Position',[100 100 1200 700]);
    for k = 1:length(alpha_list)
        target2_sigma = alpha_list(k);
        Mix = zeros(N_samples,N_chirps);
        amp1 = sqrt(target1_sigma/sigma_ref)*(R_ref/target1_R)^2;
        amp2 = sqrt(target2_sigma/sigma_ref)*(R_ref/target2_R)^2;
        tau1 = 2*target1_R/c;
        fd1  = 2*target1_v/lambda;
        tau2 = 2*target2_R/c;
        fd2  = 2*target2_v/lambda;
        t = (0:length(tx)-1)'/Fs;

        rx1 = amp1*delayseq(tx,tau1,Fs);
        rx1 = rx1 .* exp(1j*2*pi*fd1*t);
    
        rx2 = amp2*delayseq(tx,tau2,Fs);
        rx2 = rx2 .* exp(1j*2*pi*fd2*t);

        Ps = mean(abs(rx1(:)).^2);
        Pn = Ps/(10^(snr/10));
    
        noise = sqrt(Pn/2)*(randn(size(rx1))+1j*randn(size(rx1)));
        rx = rx1 + rx2;
        rx_noisy = rx + noise;
    
        mix = tx .* conj(rx_noisy);
        Mix = reshape(mix,N_samples,N_chirps);
    
        window_fast = hann(N_samples);
        window_slow = hann(N_chirps);
        window_2d   = window_fast*window_slow';
    
        N_fft_padded = 4096;
    
        R_max = Fs*c/(2*S);
        range_axis = linspace(0,R_max,N_fft_padded);
        vel_axis = linspace(v_max, -v_max, N_chirps);
        rdm = fftshift(fft2(Mix.*window_2d,N_fft_padded,N_chirps),2);
        rdm_mag = abs(rdm);
        rdm_db = 20*log10(rdm_mag + eps);
        rdm_disp = rdm_db - max(rdm_db(:));
    
        range_prof = max(rdm_db,[],2);
        [peak_r_db,idx_r] = max(range_prof);
        detected_range = range_axis(idx_r);
        dop_prof = rdm_db(idx_r,:);
        [peak_v_db,idx_v] = max(dop_prof);
        detected_velocity = vel_axis(idx_v);
        
         
         subplot(2,3,k)
         plot(vel_axis,dop_prof)
         grid on
         title(sprintf('\\alpha = %.2f',target2_sigma))
         xlabel('Velocity (m/s)')
         ylabel('Magnitude (dB)')
    end

    sgtitle(sprintf('Same Range, Different Doppler (SNR=%d dB)',snr))

end

%% Same doppler different range

target1_R = 30;
target1_v = 10;
target1_sigma = 1;
Range_list = [10 15 20 50 60];


target2_v = 10; %SAME DOPPLER
target2_sigma = 1;
for snr = SNR_list
    figure('Name',sprintf('Same Doppler Different Range - SNR = %d dB',snr),'Position',[100 100 1200 700]);

    for k = 1:length(Range_list)
            target2_R = Range_list(k);
    
            Mix = zeros(N_samples,N_chirps);
            amp1 = sqrt(target1_sigma/sigma_ref)*(R_ref/target1_R)^2;
            amp2 = sqrt(target2_sigma/sigma_ref)*(R_ref/target2_R)^2;
            tau1 = 2*target1_R/c;
            fd1  = 2*target1_v/lambda;
            tau2 = 2*target2_R/c;
            fd2  = 2*target2_v/lambda;
            t = (0:length(tx)-1)'/Fs;
    
            rx1 = amp1*delayseq(tx,tau1,Fs);
            rx1 = rx1 .* exp(1j*2*pi*fd1*t);
        
            rx2 = amp2*delayseq(tx,tau2,Fs);
            rx2 = rx2 .* exp(1j*2*pi*fd2*t);
    
            Ps = mean(abs(rx1(:)).^2);
            Pn = Ps/(10^(snr/10));
        
            noise = sqrt(Pn/2)*(randn(size(rx1))+1j*randn(size(rx1)));
            rx = rx1 + rx2;
            rx_noisy = rx + noise;
        
            mix = tx .* conj(rx_noisy);
            Mix = reshape(mix,N_samples,N_chirps);
        
        
            window_fast = hann(N_samples);
            window_slow = hann(N_chirps);
            window_2d   = window_fast*window_slow';
        
            N_fft_padded = 4096;
        
            R_max = Fs*c/(2*S);
            range_axis = linspace(0,R_max,N_fft_padded);
            vel_axis = linspace(v_max, -v_max, N_chirps);
            rdm = fftshift(fft2(Mix.*window_2d,N_fft_padded,N_chirps),2);
            rdm_mag = abs(rdm);
            rdm_db = 20*log10(rdm_mag + eps);
            rdm_disp = rdm_db - max(rdm_db(:));
        
            range_prof = max(rdm_db,[],2);
            [peak_r_db,idx_r] = max(range_prof);
            detected_range = range_axis(idx_r);
            dop_prof = rdm_db(idx_r,:);
            [peak_v_db,idx_v] = max(dop_prof);
            detected_velocity = vel_axis(idx_v);
    
            subplot(2,3,k)
            plot(range_axis,range_prof)
            grid on
            title(sprintf('R = %d m',target2_R))
            xlabel('Range (m)')
            ylabel('Magnitude (dB)')
    end
    sgtitle(sprintf('Same Doppler, Different Range (SNR=%d dB)',snr))
end

%% CFAR Range FFT
cfar = phased.CFARDetector( ...
    'Method', 'CA', ...             
    'NumTrainingCells', 100, ...    
    'NumGuardCells', 10, ...        
    'ProbabilityFalseAlarm', 1e-6); 
release(cfar);
cfar.ThresholdOutputPort = true;


range_prof_power = max(abs(rdm).^2, [], 2); 
[detections, threshold] = cfar(range_prof_power, 1:length(range_prof_power));
range_prof_db = 10 * log10(range_prof_power + eps);
threshold_db = 10 * log10(threshold + eps);

figure('Name', 'Range and doppler cfar overlay', 'Position', [100, 100, 1200, 500]);
subplot(1,2,1);
plot(range_axis, range_prof_db, 'b', 'LineWidth', 1.5); 
hold on;
plot(range_axis, threshold_db, 'r--', 'LineWidth', 2);


raw_idx = find(detections);
idx = raw_idx(range_prof_power(raw_idx) >= range_prof_power(raw_idx-1) & ...
              range_prof_power(raw_idx) >= range_prof_power(raw_idx+1));
plot(range_axis(idx), range_prof_db(idx), 'wo', 'MarkerFaceColor', 'r');

grid on;
title('CFAR Detection (dB)');
xlabel('Range (m)'); ylabel('Power (dB)');
legend('Range Profile', 'CFAR Threshold', 'Detections');


%% Doppler CFAR

doppler_prof_power = abs(rdm(idx_r, :)).^2; 


cfar_dop = phased.CFARDetector( ...
    'Method', 'CA', ...             
    'NumTrainingCells', 10, ...   
    'NumGuardCells', 2, ...        
    'ProbabilityFalseAlarm', 1e-6); 
release(cfar_dop);
cfar_dop.ThresholdOutputPort = true;

[detections_dop, threshold_dop] = cfar_dop(doppler_prof_power', 1:length(doppler_prof_power));

doppler_prof_db = 10 * log10(doppler_prof_power(:) + eps);
threshold_dop_db = 10 * log10(threshold_dop(:) + eps);


subplot(1,2,2)
plot(vel_axis, doppler_prof_db, 'm', 'LineWidth', 1.5); 
hold on;
plot(vel_axis, threshold_dop_db, 'b--', 'LineWidth', 2);

idx_d = find(detections_dop);
plot(vel_axis(idx_d), doppler_prof_db(idx_d), 'ro', 'MarkerFaceColor', 'r');

grid on;
title('Doppler CFAR Detection (dB)');
xlabel('Velocity (m/s)'); ylabel('Power (dB)');

%% 2D CFAR: Before and After Comparison

cfar2D = phased.CFARDetector2D(...
    'GuardBandSize', [4, 4], ...    
    'TrainingBandSize', [10, 10], ...
    'ProbabilityFalseAlarm', 1e-6);

rdm_power = abs(rdm).^2; 
[nRange, nDop] = size(rdm_power);

pad = 14; 
[colInds, rowInds] = meshgrid((pad+1):(nDop-pad), (pad+1):(nRange-pad)); 
CUTIdx = [rowInds(:), colInds(:)]'; 

temp_mask = cfar2D(rdm_power, CUTIdx);

mask_matrix = false(nRange, nDop);
for i = 1:size(CUTIdx, 2)
    mask_matrix(CUTIdx(1,i), CUTIdx(2,i)) = temp_mask(i);
end

figure('Name', '2D CFAR: Before vs. After', 'Position', [100, 100, 1200, 500]);

subplot(1, 2, 1);
imagesc(vel_axis, range_axis, 20*log10(rdm_power + eps));
axis xy; colormap jet; colorbar;
title('Before CFAR'); xlabel('Velocity (m/s)'); ylabel('Range (m)');
clim([-40 10]);

subplot(1, 2, 2);
rdm_after = 20*log10(rdm_power + eps);
rdm_after(~mask_matrix) = -100; 
imagesc(vel_axis, range_axis, rdm_after);
axis xy; colormap jet; colorbar;
title('After CFAR'); xlabel('Velocity (m/s)'); ylabel('Range (m)');
clim([-50 100]);