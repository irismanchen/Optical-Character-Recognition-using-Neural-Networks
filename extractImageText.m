function [text] = extractImageText(fname)
% [text] = extractImageText(fname) loads the image specified by the path 'fname'
% and returns the next contained in the image as a string.
load('nist36_model.mat', 'W', 'b');
im = imread(fname);
[lines, bw] = findLetters(im);
lineSize = length(lines);
result = cell(1, lineSize);
outputs = cell(1, lineSize);
testLabels = cell(1, lineSize);
l = 32;
text = '';
class = ['A' 'B' 'C' 'D' 'E' 'F' 'G' 'H' 'I' 'J' 'K' 'L' 'M'... 
         'N' 'O' 'P' 'Q' 'R' 'S' 'T' 'U' 'V' 'W' 'X' 'Y' 'Z'...
         '0' '1' '2' '3' '4' '5' '6' '7' '8' '9'];
for i = 1:lineSize
    curLine = lines{i};
    result{i} = zeros(size(curLine,1), 1024);
    for j = 1:size(curLine,1)
        tmp = bw(curLine(j, 2):curLine(j, 4),curLine(j, 1):curLine(j, 3));
        tmp = imresize(tmp, [l l]);
        result{i}(j,:) = reshape(tmp, 1,l*l);
    end
end
for i = 1:lineSize
    outputs{i} = Classify(W, b, result{i});
    [~,testLabels{i}] = max(outputs{i},[],2);
    testLabels{i} = class(testLabels{i});
    text = [text testLabels{i} char(10)];
    %text = [text testLabels{i}];
end
