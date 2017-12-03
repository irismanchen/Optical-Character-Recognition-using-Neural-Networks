function [lines, bw] = findLetters(im)
% [lines, BW] = findLetters(im) processes the input RGB image and returns a cell
% array 'lines' of located characters in the image, as well as a binary
% representation of the input image. The cell array 'lines' should contain one
% matrix entry for each line of text that appears in the image. Each matrix entry
% should have size Lx4, where L represents the number of letters in that line.
% Each row of the matrix should contain 4 numbers [x1, y1, x2, y2] representing
% the top-left and bottom-right position of each box. The boxes in one line should
% be sorted by x1 value.


%Your code here
imgray = rgb2gray(im);
bw = imbinarize(imgray);
CC = bwconncomp(~bw);
pixels = CC.PixelIdxList;
L = labelmatrix(CC);
threshold = 10;
margin = size(im,2)/200;
rects = [];
for n = 1:length(pixels)
    [rows, cols] = find(L == n);
    up = min(rows);
    down = max(rows);
    left = min(cols);
    right = max(cols);
    if (down-up<threshold+size(im, 2)/90)||(right-left<threshold)
        continue;
    end
    rects = [rects [left-margin;up-margin;right+margin;down+margin]];
end
% sort according to the top line
[~, index] = sort(rects(2,:));
rects = rects(:,index);
lineMatrix = rects(:,1);
lineNum = 1;
lines = cell(1,1);
for i = 2:size(rects, 2)
    if rects(2,i)>rects(4, i-1)
        lines{lineNum} = lineMatrix';
        lineNum = lineNum + 1;
        lineMatrix = rects(:,i);
    else
        lineMatrix = [lineMatrix rects(:, i)];
    end
end
lines{lineNum} = lineMatrix';
for i = 1:length(lines)
    curLine = lines{i}';
    [~, I] = sort(curLine(1,:));
    curLine = curLine(:,I);
    lines{i} = curLine';
end
assert(size(lines{1},2) == 4,'each matrix entry should have size Lx4');
assert(size(lines{end},2) == 4,'each matrix entry should have size Lx4');
lineSortcheck = lines{1};
assert(issorted(lineSortcheck(:,1)) | issorted(lineSortcheck(end:-1:1,1)),'Matrix should be sorted in x1');

end