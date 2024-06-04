function [] = CAP_writeExcelMetrics_TwoPop_SW(SavePath, fancyName,Outputs, nTP,mySubjects, nSubj)
    %% Extract CAPs Measures
    % Make sure you read in the data correctly: As the code is now, it
    % takes Group1 as reference population.
    % 
    % Samantha Weber June 2022

    %Extract Duration
    Duration_G1 = Outputs.Metrics.AverageExpressionDuration{1,1}(:,3:end-1); %rows are subjects, columns are states
    Duration_G1(isnan(Duration_G1))=0;

    Duration_G2 = Outputs.Metrics.AverageExpressionDuration{1,2}(:,3:end-1); %rows are subjects, columns are states
    Duration_G2(isnan(Duration_G2))=0;

    %Extract Entries
    Entries_G1 = Outputs.Metrics.NumberEntries{1,1}(:,3:end-1);
    Entries_G1(isnan(Entries_G1))=0;

    Entries_G2 = Outputs.Metrics.NumberEntries{1,2}(:,3:end-1);
    Entries_G2(isnan(Entries_G2))=0;

    %Extract Entries
    ExpressionDuration_G1 = Outputs.Metrics.AverageExpressionDuration{1,1}(:,3:end-1);
    ExpressionDuration_G1(isnan(ExpressionDuration_G1))=0;

    ExpressionDuration_G2 = Outputs.Metrics.AverageExpressionDuration{1,2}(:,3:end-1);
    ExpressionDuration_G2(isnan(ExpressionDuration_G2))=0;

    %Extract Occurences
    Occurences_G1 = Outputs.Metrics.Occurrences{1,1}.raw.state(:,1:end-1);
    Occurences_G1(isnan(Occurences_G1))=0;

    Occurences_G2 = Outputs.Metrics.Occurrences{1,2}.raw.state(:,1:end-1);
    Occurences_G2(isnan(Occurences_G2))=0;

    notpicked_G1 = Outputs.Metrics.Occurrences{1,1}.raw.notpicked(:,1);
    notpicked_G1(isnan(notpicked_G1))=0;

    scrubbed_G1 = Outputs.Metrics.Occurrences{1,1}.raw.scrubbed(:,1);
    scrubbed_G1(isnan(scrubbed_G1))=0;
    nSelected_G1 = nTP -(notpicked_G1 + scrubbed_G1);

    notpicked_G2 = Outputs.Metrics.Occurrences{1,2}.raw.notpicked(:,1);
    notpicked_G2(isnan(notpicked_G2))=0;

    scrubbed_G2 = Outputs.Metrics.Occurrences{1,2}.raw.scrubbed(:,1);
    scrubbed_G2(isnan(scrubbed_G2))=0;
    nSelected_G2 = nTP -(notpicked_G2 + scrubbed_G2);


    %Extract Resilience
    CapResilience_G1 = Outputs.Metrics.CAPResilience{1,1};
    CapResilience_G1(isnan(CapResilience_G1))=0;

    CapResilience_G2 = Outputs.Metrics.CAPResilience{1,2};
    CapResilience_G2(isnan(CapResilience_G2))=0;

    %Extract Baseline Resilience
    BaselineResilience_G1 = Outputs.Metrics.BaselineResilience{1,1};
    BaselineResilience_G1(isnan(BaselineResilience_G1))=0;

    BaselineResilience_G2 = Outputs.Metrics.BaselineResilience{1,2};
    BaselineResilience_G2(isnan(BaselineResilience_G2))=0;

    %Extract Entries from Baseline
    CAPEntriesFromBaseline_G1 = Outputs.Metrics.CAPEntriesFromBaseline{1,1};
    CAPEntriesFromBaseline_G1(isnan(CAPEntriesFromBaseline_G1))=0;

    CAPEntriesFromBaseline_G2 = Outputs.Metrics.CAPEntriesFromBaseline{1,2};
    CAPEntriesFromBaseline_G2(isnan(CAPEntriesFromBaseline_G2))=0;

    %Extract Exit to Baseline
    CAPExitToBaseline_G1 = Outputs.Metrics.CAPExitsToBaseline{1,1};
    CAPExitToBaseline_G1(isnan(CAPExitToBaseline_G1))=0;

    CAPExitToBaseline_G2 = Outputs.Metrics.CAPExitsToBaseline{1,2};
    CAPExitToBaseline_G2(isnan(CAPExitToBaseline_G2))=0;

    %Extract Transition
    Transition_G1 = Outputs.Metrics.TransitionProbabilities{1,1}(3:end-1,3:end-1,:);
    Transition_G1(isnan(Transition_G1))=0;

    Transition_G2 = Outputs.Metrics.TransitionProbabilities{1,2}(3:end-1,3:end-1,:);
    Transition_G2(isnan(Transition_G2))=0;

    %% Define Number of States

    nStates = size(Occurences_G1,2);
    namesStates = {};
    namesSubj = {};

    %CAPs header
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
    Duration =[];
    for i = 1:nStates
        thisDuration = [num2cell(Duration_G1(:,i)); num2cell(Duration_G2(:,i))];
        Duration = [Duration; thisDuration];
    end
    Duration = [header; namesSubj namesStates Duration];
    clear i thisDuration

    % Entries
    Entries = [];
    for i = 1:nStates
        thisEntry = [num2cell(Entries_G1(:,i)); num2cell(Entries_G2(:,i))];
        Entries = [Entries; thisEntry];
    end
    Entries = [header; namesSubj namesStates Entries];
    clear i thisEntry

    % Average Expression Duration
    ExpressionDuration = [];
    for i = 1:nStates
        thisExpressionDuration = [num2cell(ExpressionDuration_G1(:,i)); num2cell(ExpressionDuration_G2(:,i))];
        ExpressionDuration = [ExpressionDuration; thisExpressionDuration];
    end
    ExpressionDuration = [header; namesSubj namesStates ExpressionDuration];
    clear i thisExpressionDuration

    % Occurence

    Occurence =[];
    for i = 1:nStates
        thisOccurence = [num2cell(Occurences_G1(:,i)); num2cell(Occurences_G2(:,i))];
        Occurence = [Occurence; thisOccurence];
    end
    Occurence = [header; namesSubj namesStates Occurence];
    clear i thisOccurence


    % Relative Occurence
    Rel_Occ_raw = [Occurences_G1; Occurences_G2];
    nSelected = [nSelected_G1; nSelected_G2];
    RelativeOccurence = [];
    for i =1:nStates
        for s = 1:nSubj
            thisRelOccurence(s,1) = (Rel_Occ_raw(s,i)/nSelected(s,1));
        end
        RelativeOccurence = [RelativeOccurence; num2cell(thisRelOccurence)];
    end
    clear thisRelOccurence s
    RelativeOccurence_final = [header; namesSubj namesStates RelativeOccurence];
    clear i 


    %Resilience
    CapResilience =[];
    for i = 1:nStates
        thisCapResilience = [num2cell(CapResilience_G1(:,i)); num2cell(CapResilience_G2(:,i))];
        CapResilience = [CapResilience; thisCapResilience];
    end
    CapResilience = [header; namesSubj namesStates CapResilience];
    clear i thisCapResilience

    %Baseline Resilience
    BaselineResilience =[];
    %for i = 1:nStates
        thisBaselineResilience = [num2cell(BaselineResilience_G1); num2cell(BaselineResilience_G2)];
        BaselineResilience = [BaselineResilience; thisBaselineResilience];
    %end
    BaselineResilience = [header; namesSubj_Baseline namesBaseline BaselineResilience];
    clear i thisBaselineResilience

    %EntriesFrom Baseline
    CapEntriesBaseline =[];
    for i = 1:nStates
        thisCapEntriesBaseline = [num2cell(CAPEntriesFromBaseline_G1(:,i)); num2cell(CAPEntriesFromBaseline_G2(:,i))];
        CapEntriesBaseline = [CapEntriesBaseline; thisCapEntriesBaseline];
    end
    CapEntriesBaseline = [header; namesSubj namesStates CapEntriesBaseline];
    clear i thisCapEntriesBaseline

    %Exits to Baseline
    CapExitsBaseline =[];
    for i = 1:nStates
        thisCapExitsBaseline = [num2cell(CAPExitToBaseline_G1(:,i)); num2cell(CAPExitToBaseline_G2(:,i))];
        CapExitsBaseline = [CapExitsBaseline; thisCapExitsBaseline];
    end
    CapExitsBaseline = [header; namesSubj namesStates CapExitsBaseline];
    clear i thisCapExitsBaseline

    Transition_G1 = [num2cell(mean(Transition_G1,3))];
    Transition_G2 = [num2cell(mean(Transition_G2,3))];


    %% Write it to excel
    % Write excel file
    filename = fullfile(SavePath,['CAPsState_PCA_TwoPop' num2str(nStates) 'CAPs.xlsx']);
    xlswrite(filename,Duration,'Duration')
    xlswrite(filename,Occurence,'Occurence')
    xlswrite(filename,RelativeOccurence_final,'RelativeOccurence');
    xlswrite(filename,Entries,'Entries')
    xlswrite(filename,CapResilience,'CapResilience')
    xlswrite(filename,BaselineResilience,'BaselineResilience')
    xlswrite(filename,ExpressionDuration,'ExpressionDuration')
    xlswrite(filename,CapEntriesBaseline,'EntriesFromBaseline')
    xlswrite(filename,CapExitsBaseline,'ExitsToBaseline')
    xlswrite(filename,Transition_G1,'Transition_Group1')
    xlswrite(filename,Transition_G2,'Transition_Group2')

end