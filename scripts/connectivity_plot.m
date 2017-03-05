i = 1; 

cmap = colormap(parula(6)); 

c_ratio = csvread('512_5k.csv'); 
hold on;
plot(c_ratio(:, 1), c_ratio(:, 2), 'LineWidth',2, 'Color', cmap(i, :))
i = i + 1; 

c_ratio = csvread('256_5k.csv'); 
hold on;
plot(c_ratio(:, 1), c_ratio(:, 2), 'LineWidth',2, 'Color', cmap(i, :))
i = i + 1; 

c_ratio = csvread('128_5k.csv'); 
hold on;
plot(c_ratio(:, 1), c_ratio(:, 2), 'LineWidth',2, 'Color', cmap(i, :))
i = i + 1; 

c_ratio = csvread('64_5k.csv'); 
hold on;
plot(c_ratio(:, 1), c_ratio(:, 2), 'LineWidth',2, 'Color', cmap(i, :))
i = i + 1; 

c_ratio = csvread('32_5k.csv'); 
hold on;
plot(c_ratio(:, 1), c_ratio(:, 2), 'LineWidth',2, 'Color', cmap(i, :))
i = i + 1; 

c_ratio = csvread('16_5k.csv'); 
hold on;
plot(c_ratio(:, 1), c_ratio(:, 2), 'LineWidth',2, 'Color', cmap(i, :))
i = i + 1; 

% c_ratio = csvread('8_5k.csv'); 
% hold on;
% plot(c_ratio(:, 1), c_ratio(:, 2), 'LineWidth',2, 'Color', cmap(i, :))
% i = i + 1; 


xlim([0, 0.6])
ylim([0, 1.0])

legend('N=512', 'N=256', 'N=128', 'N=64', 'N=32', 'N=16')

xlabel('ratio of (randomly) connected node pairs');
ylabel('p(connected)')