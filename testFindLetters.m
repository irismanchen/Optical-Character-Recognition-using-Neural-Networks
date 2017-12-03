% Your code here.
images = cell(1, 4);
images{1} = imread('../images/01_list.jpg');
images{2} = imread('../images/02_letters.jpg');
images{3} = imread('../images/03_haiku.jpg');
images{4} = imread('../images/04_deep.jpg');

for i = 1:4
    clf;
    [lines, bw] = findLetters(images{i});
    figure(1);
    imshow(images{i});
    hold on;
    for j = 1:length(lines)
        curLine = lines{j}';
        for k =1:size(curLine, 2)
            rectangle('Position', [curLine(1,k) curLine(2,k) curLine(3,k)-curLine(1,k) curLine(4,k)-curLine(2,k)],...
                'EdgeColor', 'r', 'LineWidth', 1);
        end
    end
    pause;
end