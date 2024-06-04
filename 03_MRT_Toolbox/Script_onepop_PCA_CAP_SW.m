%% PCA CAPs - One Populations
% This is the example script to run the PCA CAPs analyses (without the GUI).
%
% In this script, we assume one population of subjects. You can have two
% groups within this "Group 1" though. It will calculate the CAPs across
% all your subjects (even though if different populations are present)
%
% Please make sure to prepare your data accordingly as reported in the
% original toolbox paper Bolton et al., TbCAPs: A toolbox for
% co-activation pattern analysis. NeuroImage 2020.
%
% You have to enter your specific data (paths, prefixes etc.) in lines 48 - 54. 
% If done correctly, the code should run without further problems. 
%
% Samantha Weber, samantha.weber@insel.ch
% v1.0 June 2022
%
% If you use this script please cite: 
% Weber et al., "Transient resting-state salience-limbic co-activation 
% patterns in functional neurological disorders", JOURNAL, YEAR, doi: 
% AND 
% Bolton et al., TbCAPs: A toolbox for co-activation pattern analysis.
% NeuroImage 2020. doi: 10.1016/j.neuroimage.2020.116621 

clear; clc;close;
addpath(genpath(pwd));

%% 1. Loading the data files
% Add here your path to your data. The folder must contain a folder called
% Group_1. You can have different populations within this folder. It will
% just calculate the CAPs across all your subjects. 

RootPath = ' DATA PATH '; % Add here the path to where you save your data. 
% Your data has to be saved in one subfolder
% called "Group_1". Within this folder, you have to have
% your subject folders which are all of same length (e.g., P001, P002
% etc.). Inside of each subject's folder you have no further subfolders.
% Inside of each subject's folder you have your final preprocessed
% functional images, and your rp* file which gets output from your
% preprocessing (realignment). We perform scrubbing here, if no rp* file is
% found, no scrubbing is performed. 

fancyName = 'OnePopulation_PCA'; %Name of your output file
SavePath = fullfile(RootPath, 'Save'); %Where your output will be safed
mkdir(fullfile(SavePath,fancyName));

%% 1.1 Define your variables
nTP = ; %Number of functional volumes
TR = ;% The TR of your data in seconds
folder_prefix = 'P*'; %prefix of your individual imaging folders, they should start all with the same letter. 
prefix = 's5w*'; %prefix of your preprocessed imaging data
seedName = {'Example_mask.nii'}; % This is your ROI. The Roi has to be a binary nifti file
MasksPath = fullfile(pwd,'\Masks'); % Add correct Path
selectMask = 'GM'; %GM %Whole-brain %WM %CSF %GM_Mask_Population
% Caveat: The GM_Mask_Population is specific for your data and is created
% using the segmented and normalized c1 images. You can find an example
% script in the DefaultData folder > MakePopulationMask.m

%% 1. Prepare file locations
Group = {'Group_1'};
n_datasets = 1;

% load Subject folder;
filelist=dir(fullfile(RootPath,Group{1},folder_prefix));
N_Subj{1}=size(filelist,1);
mySubj=cell(1,N_Subj{1});
for f=1:N_Subj{1}
    mySubj{f}=filelist(f).name;
end
clear filelist f

% Define the folders
for i = 1:N_Subj{1}
    functdir{i} = fullfile(RootPath, Group{1},mySubj{i});
end

%% 1.1 Loading the data files
% This takes very long and needs a lot of RAM. On a "normal" computer you
% can load around 20/20 subjects, but if you want more, better run it on a
% server. 
mask ={};
brain_info ={};
n_dataset = 0;

% Mask: n_voxels x 1 logical vector
% Header: the header (obtained by spm_vol) of one NIFTI file with proper
% data dimension and .mat information
[mask, brain_info] = createMask_CAP_SW(functdir, prefix, selectMask);
Tstart = clock;
% Data: cell array, each cell of size n_TP x n_masked_voxels
TC ={};
FD ={};
[TC,brain,FD] = loadData_SW(functdir, prefix, mask, brain_info);
FD = FD{:,:};
TC = TC{:,:};

%Tstart = clock;
Ttotal=etime(clock, Tstart);
disp(['** Subjects loaded. Total time: ' num2str(Ttotal/60,'%3.1f') ' min.']);

% Load seed
disp(' ');
disp('-----------------------------------------------------');
disp('Load seed...');
[Seed] = loadseed_SW(functdir, prefix, mask, brain_info, seedName,MasksPath);

% Computes seed maps for each subject and for the population, using the
% data from the chosen reference population
[~,AvgSeedMap] = CAP_Compute_SeedMap(TC,Seed,1);


%% 2. Specifying the main parameters

% Threshold above which to select frames
T = 1; %more restricted = 0.84, CDF-1(0.80) = 0.84)

% Selection mode ('Threshold' or 'Percentage')
SelMode = 'Threshold';

% Threshold of FD above which to scrub out the frame and also the t-1 and
% t+1 frames (if you want another scrubbing setting, directly edit the
% code)
Tmot = 0.5;

% Type of used seed information: select between 'Average','Union' or
% 'Intersection'
SeedType = 'Average'; %'Average';

% Contains the information, for each seed (each row), about whether to
% retain activation (1 0) or deactivation (0 1) time points
switch SeedType 
    case 'Union'
        Activation = [1,0];
        for s = 1:size(seedName,2)
            SignMatrix(s,:) = Activation; 
        end
    case 'Average'
        SignMatrix = [1 0];
end

% Percentage of positive-valued voxels to retain for clustering
Pp = 100;

% Percentage of negative-valued voxels to retain for clustering
Pn = 100;

% Number of repetitions of the K-means clustering algorithm
n_rep = 80;

% Percentage of frames to use in each fold of consensus clustering
Pcc = 80;

% Number of folds we run consensus clustering for
N = 80;

%% 3. Selecting the frames to analyse    

% Xon will contain the retained frames, and Indices will tag the time
% points associated to these frames, for each subject (it contains a
% subfield for retained frames and a subfield for scrubbed frames)
[Xon,p,Indices,idx_sep_seeds,Xonp_scrub] = CAP_find_activity(TC,Seed,T,FD,Tmot,SelMode,SeedType,SignMatrix);
   
% Percentage of retained frames across subjects
RetainedPercentage{1} = p(3,:);

% Indices of the frames that have been retained (used later for metrics
% computations)
FrameIndices{1} = Indices; 
tmp_toplot = ConcatMat(RetainedPercentage,1,1,N_Subj,'FD');

% Displays the violin plot of subject scrubbing percentage for the
% reference population
TPViolin = figure;
axes1 = axes(TPViolin);

% Colors used in plotting of all populations
PopColor{1} = [255,255,180; 219,224,252; 188,252,188; 230,230,230]/255;
PopColor{2} = [130,48,48; 51,75,163; 59,113,86; 0, 0, 0]/255;

%Create plot
[~,~,TPViolin] = MakeViolin_SW(tmp_toplot,axes1,{' '},'Frames ret. [%]',PopColor,1,1);

%Save plot as .jpg and .fig 
saveas(gcf,fullfile(SavePath,fancyName,'RetainedFrames.jpg'));
saveas(gcf,fullfile(SavePath,fancyName,'RetainedFrames.fig'));
disp('-----------------------------------------------------');
disp(' ');
disp(['Figure Frames retained has been saved in ' SavePath]);
close;
%% 4. Consensus clustering (if wished to determine the optimum K)

Xon1_pca = [];

for i = 1:N_Subj{1,1}

    Xon1_pca = [Xon1_pca, Xon{1,i}];
    
end

[U,W,Eigenvals,mu] = ComputePCA_SW(Xon1_pca);

% After the call, U will have size V x T, and W will have size T x T. Be
% careful that in our function, row i of W contain the weights, for all
% T frames, associated to the principal direction i (contained as the i-th
% column in U).

% Also, notice that we want to compute only one PCA on the
% population-wise data, not one per subject! Otherwise, we would have a
% different dimensionality reduction for each subject, which would make our
% life complicated...

% Later, you want to input "{W}" (rows = dimensions, columns = data points) to
% CAP_ConsensusClustering. The brackets are because the function wants a
% cell input...

% After the consensus clustering step, you will want to feed W to your
% k-means clustering as well instead of "cell2mat(Xon1)". Your output CAP
% matrix will have size K x T, with K the number of CAPs. In order to go
% back to the original space, you will apply:
%               CAP_original = (U*CAP')'
% This will give you CAPs with a dimensionality K x V, i.e., what you would
% have obtained using CAP analysis without PCA


%% Consensus clustering
% This specifies the range of values over which to perform consensus
% clustering: if you want to run parallel consensus clustering processes,
% you should feed in different ranges to each call of the function
Tstart = clock;
K_range = 2:20;

% Have each of these run in a separate process on the server =)
disp('Running Consensus Clustering...')

[Consensus] = CAP_ConsensusClustering({W},K_range,'items',Pcc/100,N,'correlation');

%Plot Consensus Matrix
ConsensusMatrixPlot_SW(Consensus,SavePath,fancyName, K_range)

% Calculates the quality metrics
[CDF,PAC] = ComputeClusteringQuality(Consensus,K_range);

disp(' ');
disp('Consensus Cluster successfully performed!');

Ttotal=etime(clock, Tstart);
disp(['** Consensus clustering completed. Total time: ' num2str(Ttotal/60,'%3.1f') ' min.']);

% Qual should be inspected to determine the best cluster number(s)
CCPlot = figure;
ax1 = axes(CCPlot);

tmp_plot = bar(2:K_range(end),1-PAC);
xlabel(get(tmp_plot(1),'Parent'),'Cluster number K');
ylabel(get(tmp_plot(1),'Parent'),'Stability');
xlim(get(tmp_plot(1),'Parent'),[2-0.6,K_range(end)+0.6]);
ylim(get(tmp_plot(1),'Parent'),[0,1]);
set(get(tmp_plot(1),'Parent'),'Box','off');
custom_cm = cbrewer('seq','Reds',25);
colormap(CCPlot,custom_cm(6:25,:));
    
saveas(gcf,fullfile(SavePath,fancyName,'Stability.jpg'));
saveas(gcf,fullfile(SavePath,fancyName,'Stability.fig'));
disp(' ');
disp('-----------------------------------------------------');
disp(['Figure stability measures has been saved in ' SavePath]);
disp(' ');

disp('Please check stability measures');

% You should fill this with the actual value 
%K_opt = 3;
prompt = 'What is the optimal cluster size? ';
K_opt = input(prompt);
close;

%% 5. Clustering into CAPs
  
[CAP,Disp,Std_Clusters,idx,CorrDist,sfrac] = Run_Clustering_Sim(W,...
        K_opt,mask,brain_info,Pp,Pn,n_rep,idx_sep_seeds,SeedType);    

    % idx will contain x elements (the index of the cluster to which the
    % considered datapoint belongs

%Plot
% Computation of the similarity
SimMat = corr(CAP',CAP');
SimMat(isnan(SimMat))=0;

% Similarity Plot
imagesc(SimMat);

tmp_cb2 = cbrewer('div','RdBu',1000);
tmp_cb2(tmp_cb2 < 0) = 0;

colormap(flipud(tmp_cb2));

saveas(gcf,fullfile(SavePath,fancyName,'Similarity.jpg'));
saveas(gcf,fullfile(SavePath,fancyName,'Similarity.fig'));
disp(' ');
disp('-----------------------------------------------------');
disp(['Figure Similarity matrix has been saved in ' SavePath]);
disp(' ');

close;

switch SeedType
    case 'Union'
        %Pie plot
           % Graph displays are stored in a common tmp_sfrac cell array
           tmp_sfrac = figure;    

           % The pie charts for each cluster are created
            for cc = 1:K_opt

                subplot(K_opt,1,cc)% nrows, ncols, nPlots
                % Pie charts
                pie(realmin*ones(size(sfrac,3),1)+squeeze(mean(sfrac(:,cc,:),1)));
                caxis([1,5]);
            end

            if size(seedName,2) == 1
                labels = {'S1'};
            elseif size(seedName,2) == 2
                labels = {'S1', 'S2', 'S1+S2'};
            elseif size(seedName,2) == 3
                labels = {'S1', 'S2', 'S3','S1+S2','S2+S3','S1+S3','S1+S2+S3'};
            end
        colormap(tmp_sfrac,(custom_cm));
        Lgnd = legend(labels);
        Lgnd.Position(1) = 0.1;
        Lgnd.Position(2) = 0.7;

        saveas(gcf,fullfile(SavePath,fancyName,'SeedFraction.jpg'));
        saveas(gcf,fullfile(SavePath,fancyName,'SeedFraction.fig'));
        disp(' ');
        disp('-----------------------------------------------------');
        disp(['Figure Seed Fraction has been saved in ' SavePath]);
        disp(' ');

        close;
end

%% 5.1. PCA Reconstruction

% To create the Nifti files, we have to reconstruct our images. 
CAP_original = (U*CAP') + mu;
CAP = CAP_original';

%% 6. Computing metrics

% % Will contain the metrics
% % State matrix (n_subjects x n_time points)
% TPM = {};
% % State counts (raw and frac)
% Counts = {};
% % Number of times entering a state
% Number = {};
% % Average duration within a state
% Avg_Duration = {};
% % Duration of all the excursions within a state
% Duration = {};
% % New more recently introduced graph theoretical metrics
% From_Baseline = {};
% To_Baseline = {};
% Baseline_resilience = {};
% Resilience = {};
% Betweenness = {};
% kin = {};
% kout = {};
% % Transition probabilities 
% TM = {};
% % Cumulative sum of states
% TPMCum = {};
% % Number of each type of frame per subject
% SubjectEntries = {};

[ExpressionMap,Counts,Entries,Avg_Duration,Duration,TransitionProbabilities,...
    From_Baseline,To_Baseline,Baseline_resilience,Resilience,Betweenness,...
    InDegree,OutDegree,SubjectEntries] = Compute_Metrics_simpler(idx,...
    Indices.kept.active,Indices.scrubbedandactive,K_opt,TR);

%Plot Transition matrix for all subjects together
tmp_toplot = squeeze(mean(TransitionProbabilities,3));
tmp_toplot = tmp_toplot(3:end-1,3:end-1);

% Make graph visible and plotting
TMGraph=imagesc(tmp_toplot);
tmp_cb = cbrewer('seq','Greys',1000);
colormap(flipud(tmp_cb));

clear tmp_toplot

% Arbitrary setting of probability scale from 0 to 0.03
caxis([0 0.03]);
axis('square','on');
axis('off');

saveas(gcf,fullfile(SavePath,fancyName,'TransitionProbabilityMatrix.jpg'));
saveas(gcf,fullfile(SavePath,fancyName,'TransitionProbabilityMatrix.fig'));
close; 

disp(' ');
disp('-----------------------------------------------------');
disp(['Figure Transition Probability Matrix has been saved in ' SavePath]);
disp(' ');

% Dynamic state plotting

% Makes the graph visible
% Concatenates information from the different datasets
tmp_toplot = [];

tmp_toplot = [tmp_toplot; ExpressionMap; 0*ones(5,size(TC{1},1))];
tmp_toplot = tmp_toplot(1:end-5,:);

custom_cm = cbrewer('qual','Set1',K_opt+1);
custom_cm = [0.05,0.05,0.05;1,1,1;custom_cm];

%Plot it
imagesc(tmp_toplot);
colormap((custom_cm));
xlabel('Time [s]');
ylabel('Subjects [-]');
caxis([-1,K_opt+1]);

clear tmp_toplot
L = line(ones(4),ones(4), 'LineWidth',2); % generate line 
   if K_opt == 3
        lgd = legend('CAP1','CAP2','CAP3','Unassigned');
   elseif K_opt == 2
         lgd = legend('CAP1','CAP2','Unassigned');
   elseif K_opt == 3
        lgd = legend('CAP1','CAP2','CAP3','Unassigned'); 
   elseif K_opt == 4
        lgd = legend('CAP1','CAP2','CAP3','CAP4','Unassigned');
   elseif K_opt == 5
        lgd = legend('CAP1','CAP2','CAP3','CAP4','CAP5','Unassigned');
   elseif K_opt == 6
        lgd = legend('CAP1','CAP2','CAP3','CAP4','CAP5','CAP6','Unassigned');
   elseif K_opt == 7
        lgd = legend('CAP1','CAP2','CAP3','CAP4','CAP5','CAP6','CAP7','Unassigned');
   % More than 7 CAPs is rather unlikely, but if so, you have to increase
   % here. Could be coded nicer, too lazy though. 
   end
   
lgd.FontSize = 10;
lgd.Position(1) = 0.65;
lgd.Position(2) = 0.5;

saveas(gcf,fullfile(SavePath,fancyName,'DynamicStates.jpg'));
saveas(gcf,fullfile(SavePath,fancyName,'DynamicStates.fig'));
close;

disp(' ');
disp('-----------------------------------------------------');
disp(['Figure Dynamic States has been saved in ' SavePath]);
disp(' ');


% Plot: Cumulative state distributions
for s = 1:size(CAP,1)
    % Cumulative distribution for the state that we want to be displayed (i.e.
    % the state from the popup menu)
    %handles.TPMCum{i} = cumsum(TPM{i} == 1; % 1 = cap --> REPLACE LATER WITH CAP, TPM = ExpressionIndices

    TPMCum{s} = cumsum(ExpressionMap == s,2); % 1 = cap --> REPLACE LATER WITH CAP, TPM = ExpressionIndices

    % Average of the considered state across subjects
    tmp_TPMCum{s} = mean(TPMCum{s},1);


    % Similarly as above, we plot time if we have a valid TR; else, we plot

    subplot(size(CAP,1),1,s)

    for j = 1:size(TPMCum{s},1)
        plot(((1:size(ExpressionMap,2))-1)*TR,...
            TPMCum{s}(j,:),...
            'Color',PopColor{1}(1,:));
        hold('on');
    end


   plot(((1:size(ExpressionMap,2))-1)*TR,...
        tmp_TPMCum{1},...
        'LineWidth',2,'Color',PopColor{2}(1,:));
    xlabel('Time [s]','FontSize',10);
    xlim([0,(size(ExpressionMap,2)-1)*TR]);

    ylabel('Cumul. sum [-]','FontSize',10);
end

saveas(gcf,fullfile(SavePath,fancyName,'CummulativeSum.jpg'));
saveas(gcf,fullfile(SavePath,fancyName,'CummulativeSum.fig'));
close;

disp(' ');
disp('-----------------------------------------------------');
disp(['Figure Cummulative Sum has been saved in ' SavePath]);
disp(' ');

%% 7. Save Metrics
% General information on the project
OverallInfo.ProjectTitle = fancyName;
OverallInfo.NumberTimePoints = size(TC{1},1);
OverallInfo.NumberVoxels = size(TC{1},2); 
OverallInfo.NumberSubjects = N_Subj; 
OverallInfo.TR = TR; 

Parameters.Inputs.DataHeader = brain_info;
Parameters.Inputs.Mask = mask;
Parameters.Inputs.Seeds = Seed;
Parameters.SpatioTemporalSelection.IsSeedFree = 0; 
Parameters.SpatioTemporalSelection.NumberSeeds = size(seedName,2);
Parameters.SpatioTemporalSelection.TypeEventRetainedPerSeed = SignMatrix;
Parameters.SpatioTemporalSelection.SeedType = SeedType;
Parameters.SpatioTemporalSelection.MotionThreshold = Tmot;
Parameters.SpatioTemporalSelection.SelectionMode = SelMode;
Parameters.SpatioTemporalSelection.FrameSelectionParameter = T;
Parameters.KMeansClustering.IsConsensusRun = 1;
Parameters.KMeansClustering.MaxClusterNumber = K_range(end); 
Parameters.KMeansClustering.PercentageDataPerFold = Pcc; 
Parameters.KMeansClustering.NumberRepetitions = n_rep;
Parameters.KMeansClustering.NumberClusters = K_opt; 
Parameters.KMeansClustering.PercentagePositiveValuedVoxelsClustered = Pp;
Parameters.KMeansClustering.PercentageNegativeValuedVoxelsClustered = Pn;

Outputs.SpatioTemporalSelection.RetainedFramesPerSeed = idx_sep_seeds;
Outputs.SpatioTemporalSelection.Indices=Indices;
Outputs.SpatioTemporalSelection.PercentageRetainedFrames = RetainedPercentage;
Outputs.SpatioTemporalSelection.AverageCorrelationMap = AvgSeedMap;
Outputs.KMeansClustering.ConsensusQuality = PAC; 
Outputs.KMeansClustering.CoActivationPatternsDispersion = Disp;
Outputs.KMeansClustering.CoActivationPatterns = CAP;
Outputs.KMeansClustering.CoActivationPatternsZScored = CAP_Zscore(CAP);
Outputs.KMeansClustering.CoActivationPatternsSTD = Std_Clusters; 
Outputs.KMeansClustering.AssignmentsToCAPs = idx;
Outputs.Metrics.CAPExpressionIndices = ExpressionMap;
Outputs.Metrics.Occurrences = Counts;
Outputs.Metrics.NumberEntries = Entries;
Outputs.Metrics.AverageExpressionDuration = Avg_Duration;
Outputs.Metrics.AllExpressionDurations = Duration;
Outputs.Metrics.TransitionProbabilities = TransitionProbabilities;
Outputs.Metrics.FractionCAPFramesPerSeedCombination = sfrac;
Outputs.Metrics.CAPEntriesFromBaseline = From_Baseline;
Outputs.Metrics.CAPExitsToBaseline = To_Baseline;
Outputs.Metrics.CAPResilience = Resilience;
Outputs.Metrics.BaselineResilience = Baseline_resilience;
Outputs.Metrics.BetweennessCentrality = Betweenness;
%Outputs.Metrics.CAPInDegree = kin;
%Outputs.Metrics.CAPOutDegree = kout;
Outputs.Metrics.SubjectCounts = SubjectEntries;

%%Uncomment next three lines, if you want to save heavy outputs
%HeavyOutputs.SpatioTemporalSelection.ClusteredFrames = Xon; %Xonp?
%HeavyOutputs.KMeansClustering.Consensus = Consensus;

% Saves NIFTI files storing the CAPs in MNI space
ReferencePopulation = 1;
CAPToNIFTI(CAP,...
   mask{ReferencePopulation},brain_info{ReferencePopulation},...
   fullfile(SavePath,fancyName),['CAP_NIFTI_',fancyName]);

CAPToNIFTI(CAP_Zscore(CAP),...
   mask{ReferencePopulation},brain_info{ReferencePopulation},...
   fullfile(SavePath,fancyName),['CAP_NIFTI_ZScored_',fancyName]);

% Saves the different variables from the program
%save(fullfile(SavePath,fancyName),'OverallInfo','Parameters','Outputs','HeavyOutputs','brain','-v7.3');
save(fullfile(SavePath,fancyName),'OverallInfo','Parameters','Outputs','brain','-v7.3');

disp('-----------------------------------------------------');
disp('CAPs computed and saved successfully!');
disp(' ');
disp('Results are saved in: ');
disp([SavePath]);

% Write temporal metrics as excel file for further statistical analyses
CAP_writeExcelMetrics_OnePop_SW(SavePath,fancyName,Outputs,nTP,mySubj,N_Subj{1})
disp('-----------------------------------------------------');
disp('CAPs temporal metrics saved as Excel!');
disp(' ');
disp('Results are saved in: ');
disp([SavePath]);
disp(' ');
disp('-----------------------------------------------------');
disp('Visualize your data by typing CAP_SW!');

%CAP_SW

close; clear;




