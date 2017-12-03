load('nist36_model.mat', 'W', 'b')
load('../data/nist36_test.mat', 'test_data', 'test_labels')
% visulize the confusion matrix
outputs = Classify(W, b, test_data);    % should be dataNum*C
[~, gtLabels] = max(test_labels,[],2);
[~, outLabels] = max(outputs,[],2);
confusion = zeros(36,36);
datanum = size(outputs,1);
for i=1:datanum
    confusion(gtLabels(i),outLabels(i)) = confusion(gtLabels(i),outLabels(i))+1;
    if(gtLabels(i)==15&&outLabels(i)==27)
        disp(i);
    end
end
imagesc(confusion);
[test_acc, test_loss] = ComputeAccuracyAndLoss(W, b, test_data, test_labels);
layers = [32*32, 800, 36];
% load the pre_trained model
load('../data/nist26_model_60iters.mat', 'W', 'b')
%[Wrandom, brandom] = InitializeNetwork(layers);
WFirstLayer = reshape(W{1}', [32, 32, 1, 800]);
montage(mat2gray(WFirstLayer), 'Size', [20, 40]);