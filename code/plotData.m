function plotData(X, y)
figure; hold on;

pos = find(y==1);
neg = find(y==0);

plot(X(pos, 1), X(pos, 2), 'linewidth', 2, 'MarkerSize', 10, 'k+');
plot(X(neg, 1), X(neg, 2), 'linewidth', 2, 'MarkerSize', 10, 'ko', 'MarkerFaceColor', 'y');

hold off;

end
