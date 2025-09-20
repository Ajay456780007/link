data=load("Analysis/results.mat");
all_results=data.all_results;

x=zeros(length(all_results),6);
for i=1:length(all_results)
    x(i,1)=all_results(i).thd_current;
    x(i,2)=all_results(i).thd_current;
    x(i,3)=all_results(i).power_factor;
    x(i,4)=all_results(i).rms_voltage;
    x(i,5)= mean(abs(all_results(i).raw_current(:)));
    x(i,6)= mean(abs(all_results(i).raw_voltage(:)));

end
disp(length(x));
y = zeros(length(all_results), 3);
for i = 1:length(all_results)
    y(i,1) = all_results(i).Kp;
    y(i,2) = all_results(i).Ki;
    y(i,3) = all_results(i).Gain;
end
disp(length(y));
hiddenLayerSize = 10; 
net = fitnet(hiddenLayerSize, 'trainlm');
net.trainParam.epochs = 500;
net.divideParam.trainRatio = 0.7; 
net.divideParam.valRatio = 0.15;  
net.divideParam.testRatio = 0.15; 

[xn, xSettings] = mapminmax(x', 0, 1); % features × samples
[yn, ySettings] = mapminmax(y', 0, 1); % labels × samples

[net, tr] = train(net, xn, yn);


save('trained_ANN_model.mat', 'net');


