% Your code here.
images = {'../images/01_list.jpg', '../images/02_letters.jpg', '../images/03_haiku.jpg', '../images/04_deep.jpg'};
gtruth = cell(4,1);
gtruth{1} = 'TODOLIST1MAKEATODOLIST2CHECKOFFTHEFIRSTTHINGONTODOLIST3REALIZEYOUHAVEALREADYCOMPLETED2THINGS4REWARDYOURSELFWITHANAP';
gtruth{2} = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890';
gtruth{3} = 'HAIKUSAREEASYBUTSOMETIMESTHEYDONTMAKESENSEREFRIGERATOR';
gtruth{4} = 'DEEPLEARNINGDEEPERLEARNINGDEEPESTLEARNING';
texts = cell(1,4);
totalChar = 0;
correctChar = 0;
for i = 1:4
    texts{i} = extractImageText(images{i});
    disp(texts{i});
    %correctChar = sum(texts{i}==gtruth{i});
    %disp(correctChar/length(texts{i}));
end
