function g = sigmoid(z)

g = zeros(size(z));
for i = 1:size(z, 1)
    for j = 1:size(z, 2)
        g(i, j) = 1 / (1 + e ^ -z(i, j));
end

end
