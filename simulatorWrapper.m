clc; close all; clear all;
%% Load data need to simulate
load("Data0001.mat")
% robot_loc = readmatrix('data000.csv');

% robot_x = robot_loc(:,2);
% robot_y = robot_loc(:,3);
% robot_z = robot_loc(:,4);
%% Load data
% CorrectInput = TrialData.Params.CorrectInput{TrialData.TargetID};
% WayPoint = TrialData.Params.Waypoint{TrialData.TargetID};
% location = find(WayPoint ~= 0);
% figure; plot3(WayPoint(:,1), WayPoint(:,2), WayPoint(:,3))

CorrectDecode = TrialData.CorrectDecode; % CorrectDecode miss state 1 and 2
TaskState = TrialData.TaskState;
FilteredClickState = TrialData.FilteredClickerState;
TargetID = TrialData.TargetID;
Waypoint = TrialData.Params.Waypoint{TargetID};
BlockTarget = TrialData.Params.CorrectInput{TargetID};
SmoothedNeuralFeatures = TrialData.SmoothedNeuralFeatures;
BroadBandData = TrialData.BroadbandData;
FilterBank = TrialData.Params.FilterBank;
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
    'PassbandFrequency',25,'PassbandRipple',0.2, ...
    'SampleRate',1e3);

LSTMFunctionName = 'net_bilstm_robot_20220824B';%'net_bilstm_robot_20220928';%;%net_bilstm_robot_20220824_early_stop
LSTM = load(LSTMFunctionName);
LSTM = LSTM.net_bilstm_robot_20220824B;%net_bilstm_robot_20220824_early_stop

%% Initialize Data
LSTMBufferSize = 1000;
LSTMBuffer = 1e-5*randn(128,LSTMBufferSize);
LSTMFeatures = 1e-5*randn(LSTMBufferSize/10,256);
biLSTMSoftMaxThresh = 0.4;
pred_series = [];

%% Preprocess the needed params
num_block = size(BlockTarget,2);
change_idx = [1 findchangepts(CorrectDecode, 'MaxNumChanges', num_block) size(CorrectDecode,2)];
block_correct_needed = zeros(size(BlockTarget));
for j = 1:size(block_correct_needed,2)
    block_correct_needed(j) = size(find(FilteredClickState(change_idx(j):change_idx(j+1)) == BlockTarget(j)),2);
end

%% For simulator go through pipeline
for j =1:size(BroadBandData,2)
    samps = size(BroadBandData{j},1);
    if samps > LSTMBufferSize
        sampes = LSTMBufferSize;
    end

    LSTMBuffer = circshift(LSTMBuffer, -samps,2);
    LSTMBuffer(:,(end-samps+1):end) = BroadBandData{j}((end-samps+1):end,:)';

    filtered_data=zeros(size(LSTMBuffer',1),size(LSTMBuffer',2),8);
    for j=9:16%hg features
        filtered_data(:,:,j) =  ((filter(...
            FilterBank(j).b, ...
            FilterBank(j).a, ...
            LSTMBuffer')));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));

    % low pass filtering
    tmp_lp = filter(lpFilt,LSTMBuffer');

    % down sampling
    tmp_hg = resample(tmp_hg,LSTMBufferSize/10,LSTMBufferSize)*5e2;
    tmp_lp = resample(tmp_lp,LSTMBufferSize/10,LSTMBufferSize);

    % removing errors in the data
    I = abs(tmp_hg>15);
    I = sum(I);
    [aa bb]=find(I>0);
    tmp_hg(:,bb) = 1e-5*randn(size(tmp_hg(:,bb)));

    I = abs(tmp_lp>15);
    I = sum(I);
    [aa bb]=find(I>0);
    tmp_lp(:,bb) = 1e-5*randn(size(tmp_lp(:,bb)));

    % normalizing between 0 and 1
    tmp_hg = (tmp_hg - min(tmp_hg(:)))/(max(tmp_hg(:))-min(tmp_hg(:)));
    tmp_lp = (tmp_lp - min(tmp_lp(:)))/(max(tmp_lp(:))-min(tmp_lp(:)));

    % concatenating into LSTM features
    LSTMFeatures = [tmp_hg tmp_lp]';

    pred =  predict(LSTM,LSTMFeatures,'ExecutionEnvironment','gpu');
    %         pred = pred +  [0.1, 0, -0.3, 0, 0, 0, 0];
    pred_series = [pred_series; pred];
    [aa bb]=max(pred);
    if aa >=  biLSTMSoftMaxThresh
        Click_Decision = bb;
        Click_Distance = aa;
    else
        Click_Decision = 0;
        Click_Distance = 0;
    end
    %disp(['LSTM o/p '  num2str(Click_Decision)])


end
%% Evaluation

% segment = zeros(size(CorrectDecode));
% segment(1) = BlockTarget(1);
% segment(change_idx) = BlockTarget(2:end);
plot(CorrectDecode)
hold on
plot(FilteredClickState)
% overall correct rate
correct_bin = CorrectDecode == FilteredClickState;
overall_correct = size(find(correct_bin),2);
overall_correct_rate = overall_correct/size(CorrectDecode,2);

correction_one = FilteredClickState(CorrectDecode == 1);
correction_one_rate = size(find(correction_one == 1),2)/size(find(CorrectDecode == 1),2);
correction_two = FilteredClickState(CorrectDecode == 2);
correction_two_rate = size(find(correction_two == 2),2)/size(find(CorrectDecode == 3),2);
correction_three = FilteredClickState(CorrectDecode == 3);
correction_three_rate = size(find(correction_three == 3),2)/size(find(CorrectDecode == 3),2);
correction_four = FilteredClickState(CorrectDecode == 4);
correction_four_rate = size(find(correction_four == 1),4)/size(find(CorrectDecode == 4),2);
correction_five = FilteredClickState(CorrectDecode == 5);
correction_five_rate = size(find(correction_five == 5),2)/size(find(CorrectDecode == 5),2);
correction_six = FilteredClickState(CorrectDecode == 6);
correction_six_rate = size(find(correction_six == 6),2)/size(find(CorrectDecode == 6),2);
correction_seven = FilteredClickState(CorrectDecode == 7);
correction_seven_rate = size(find(correction_seven == 7),2)/size(find(CorrectDecode == 7),2);

seperate_correction = [correction_one_rate,correction_two_rate,correction_three_rate,correction_four_rate,correction_five_rate,correction_six_rate,correction_seven_rate];

for j = 1:size(seperate_correction,2)
    tmp = seperate_correction(j);
    if isnan(tmp) | isinf(tmp)
        seperate_correction(j) = 0;
    end
end
figure
plot(seperate_correction)
