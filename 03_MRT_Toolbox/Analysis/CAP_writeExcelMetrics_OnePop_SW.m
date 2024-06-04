function [] = CAP_writeExcelMetrics_OnePop_SW(SavePath, fancyName,Outputs, nTP,mySubjects, nSubj)

%% Extract CAPs Measures

%Extract Duration
Duration = Outputs.Metrics.AverageExpressionDuration(:,3:end-1); %rows are subjects, columns are states
Duration(isnan(Duration))=0;

%Extract Entries
Entries = Outputs.Metrics.NumberEntries(:,3:end-1);
Entries(isnan(Entries))=0;

%Extract Entries
ExpressionDuration = Outputs.Metrics.AverageExpressionDuration(:,3:end-1);
ExpressionDuration(isnan(ExpressionDuration))=0;

%Extract Occurences
Occurences = Outputs.Metrics.Occurrences.raw.state(:,1:end-1);
Occurences(isnan(Occurences))=0;

notpicked = Outputs.Metrics.Occurrences.raw.notpicked(:,1);
notpicked(isnan(notpicked))=0;
scrubbed = Outputs.Metrics.Occurrences.raw.scrubbed(:,1);
scrubbed(isnan(scrubbed))=0;
nSelected = nTP -(notpicked + scrubbed);

%Extract Resilience
CapResilience = Outputs.Metrics.CAPResilience;
CapResilience(isnan(CapResilience))=0;

%Extract Baseline Resilience
BaselineResilience = Outputs.Metrics.BaselineResilience;
BaselineResilience(isnan(BaselineResilience))=0;

%Extract Entries from Baseline
CAPEntriesFromBaseline = Outputs.Metrics.CAPEntriesFromBaseline;
CAPEntriesFromBaseline(isnan(CAPEntriesFromBaseline))=0;

%Extract Exit to Baseline
CAPExitToBaseline = Outputs.Metrics.CAPExitsToBaseline;
CAPExitToBaseline(isnan(CAPExitToBaseline))=0;

%Extract Transition
Transition = Outputs.Metrics.TransitionProbabilities(3:end-1,3:end-1,:);
Transition(isnan(Transition))=0;

%% Define Number of States

nStates = size(Occurences,2);
namesStates = {};
namesSubj = {};

for i = 1:nStates
    for s = 1:nSubj
        thisState{s,1} = ['CAP ' num2str(i)];
    end
        namesStates = [namesStates; thisState];
        namesSubj = [namesSubj; mySubjects'];
end
clear thisState i s

header = {'ID','State','value'};

%Baseline header

for s = 1:nSubj
    namesBaseline{s,1} = 'Baseline';
end
namesSubj_Baseline = mySubjects';

clear  s

%% Merge it all together to export to excel
% Output will be an excel file in columns

% Duration
Duration_new=[];
for i = 1:nStates
    thisDuration = num2cell(Duration(:,i));
    Duration_new = [Duration_new; thisDuration];
end
Duration_final = [header; namesSubj namesStates Duration_new];
clear i thisDuration

% Entries
Entries_new = [];
for i = 1:nStates
    thisEntry = num2cell(Entries(:,i));
    Entries_new = [Entries_new; thisEntry];
end
Entries_final = [header; namesSubj namesStates Entries_new];
clear i thisEntry

% Average Expression Duration
ExpressionDuration_new = [];
for i = 1:nStates
    thisExpressionDuration = num2cell(ExpressionDuration(:,i));
    ExpressionDuration_new = [ExpressionDuration_new; thisExpressionDuration];
end
ExpressionDuration_final = [header; namesSubj namesStates ExpressionDuration_new];
clear i thisExpressionDuration

% Occurence

Occurence_new =[];
for i = 1:nStates
    thisOccurence = num2cell(Occurences(:,i));
    Occurence_new = [Occurence_new; thisOccurence];
end
Occurence_final = [header; namesSubj namesStates Occurence_new];
clear i thisOccurence

% Relative Occurence

RelativeOccurence = [];
for i =1:nStates
    for s = 1:nSubj
        thisRelOccurence(s,1) = (Occurences(s,i)/nSelected(s,1));
    end
    RelativeOccurence = [RelativeOccurence; num2cell(thisRelOccurence)];
end
clear thisRelOccurence s
RelativeOccurence_final = [header; namesSubj namesStates RelativeOccurence];
clear i 


%Resilience
CapResilience_new =[];
for i = 1:nStates
    thisCapResilience = num2cell(CapResilience(:,i));
    CapResilience_new = [CapResilience_new; thisCapResilience];
end
CapResilience_final = [header; namesSubj namesStates CapResilience_new];
clear i thisCapResilience


%Baseline Resilience
BaselineResilience_new =[];
BaselineResilience_new = num2cell(BaselineResilience);

BaselineResilience_final = [header; namesSubj_Baseline namesBaseline BaselineResilience_new];
clear i thisBaselineResilience

%Entries from Baseline
CAPEntriesFromBaseline_new =[];
for i = 1:nStates
    thisCAPEntriesFromBaseline = num2cell(CAPEntriesFromBaseline(:,i));
    CAPEntriesFromBaseline_new = [CAPEntriesFromBaseline_new; thisCAPEntriesFromBaseline];
end
CAPEntriesFromBaseline_final = [header; namesSubj namesStates CAPEntriesFromBaseline_new];
clear i thisCAPEntriesFromBaseline

%Exit to Baseline
CAPExitToBaseline_new =[];
for i = 1:nStates
    thisCAPExitToBaseline = num2cell(CAPExitToBaseline(:,i));
    CAPExitToBaseline_new = [CAPExitToBaseline_new; thisCAPExitToBaseline];
end
CAPExitToBaseline_final = [header; namesSubj namesStates CAPExitToBaseline_new];
clear i thisCAPExitToBaseline

%Transition Probability
Transition = [num2cell(mean(Transition,3))];


%% Write it to excel
% Write excel file
filename = fullfile(SavePath,['CAPsState_PCA_OnePop' num2str(nStates) 'CAPs.xlsx']);
xlswrite(filename,Duration_final,'Duration')
xlswrite(filename,Occurence_final,'Occurence')
xlswrite(filename,RelativeOccurence_final, 'RelativeOccurence');
xlswrite(filename,Entries_final,'Entries')
xlswrite(filename,CapResilience_final,'CapResilience')
xlswrite(filename,BaselineResilience_final,'BaselineResilience')
xlswrite(filename,CAPEntriesFromBaseline_final,'EntriesFromBaseline')
xlswrite(filename,CAPExitToBaseline_final,'ExitsToBaseline')
xlswrite(filename,ExpressionDuration_final,'ExpressionDuration')
xlswrite(filename,Transition,'Transition')

end
