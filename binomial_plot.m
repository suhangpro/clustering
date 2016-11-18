M = 4096;
K = 100;
p = 1 / (1 + K);

xs = 0:M;
Ps = binopdf(xs, M, p);
y_lim = 0.07;

figure;
plot(xs, Ps, 'LineWidth',3);
hold on;

mu = p * M;
plot([mu, mu], [0, y_lim], 'k:', 'LineWidth',2);
hold on;

bars = [66, 80, 92, 102];

for i=1:numel(bars), 
    plot([bars(i), bars(i)], [0, y_lim], ':', 'LineWidth',2, 'Color',[(102+35*i)/255,0,0]);
    hold on;
end

xlim([15, 115])
ylim([0, 0.07])

legend('pdf', 'mean (40.5)', '1e-4 chance (66)', '1e-8 chance (80)', '1e-12 chance (92)', '1e-16 chance (102)')

xlabel('rank-1 counts');
ylabel('p')