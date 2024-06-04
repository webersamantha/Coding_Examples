%% PCA CAPs - Two Populations
% This is the example script to run the PCA CAPs analyses (without the GUI).
%
% In this script, we assume two population of subjects. Group 1 is your
% reference population. It will calculate CAPs based on your reference
% population.
%
% Please make sure to prepare your data accordingly as reported in the
% original toolbox paper Bolton et al., TbCAPs: A toolbox for
% co-activation pattern analysis. NeuroImage 2020.
%
% You have to enter your specific data (paths, prefixes etc.) in lines 46 - 53. 
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
% Group_1 and Group_2. Group_1 is your reference population

RootPath = 'DATA PATH'; % Add here the path to where you save your data. 
% Your data has to be saved in two subfolders
% called "Group_1" and "Group_2". Within those folders, you have to have
% your subject folders which are all of same length (e.g., P001, P002
% etc.). Inside of each subject's folder you have no further subfolders.
% Inside of each subject's folder you have your final preprocessed
% functional images, and your rp* file which gets output from your
% preprocessing (realignment). We perform scrubbing here, if no rp* file is
% found, no scrubbing is performed. 

fancyName = 'TwoPopulation_PCA'; %Name of your output file
SavePath = fullfile(RootPath, 'Save'); %Where your output will be safed
mkdir(fullfile(SavePath,fancyName));

%% 1.1 Define your variables
nTP = ; %Number of functional volumes
TR =   ;% The TR of your data in seconds
folder_prefix = 'P*'; %prefix of your individual imaging folders, they should start all with the same letter. 
prefix = 's5w*'; %prefix of your preprocessed imaging data
seedName = {'Example_mask.nii'}; % This is your ROI. The Roi has to be a binary nifti file
MasksPath = fullfile(pwd,'\Masks'); % Add correct Path
selectMask = 'GM'; %GM %Whole-brain %WM %CSF %GM_Mask_Population
% Caveat: The GM_Mask_Population is specific for your data and is created
% using the segmented and normalized c1 images. You can find an example
% script in the DefaultData folder > MakePopulationMask.m


%% 1. Prepare file locations
Group = {'Group_1', 'Group_2'};
n_datasets = 2;

% load healthy control folder (group 1);
filelist=dir(fullfile(RootPath,Group{1}, folder_prefix)); %P is the sufix of your individual subject folders, adapt accordingly
N_G1{1}=size(filelist,1);
myG1=cell(1,N_G1{1});
for f=1:N_G1{1}
    myG1{f}=filelist(f).name;
end
clear filelist f

% load patient folder (group 2);
filelist=dir(fullfile(RootPath,Group{2}, folder_prefix));%P is the sufix of your individual subject folders, adapt accordingly
N_G2{1}=size(filelist,1);
myG2=cell(1,N_G2{1});
for f=1:N_G2{1}
    myG2{f}=filelist(f).name;
end
clear filelist f

mySubj = [myG1, myG2];
N_Subj{1} = N_G1{1} + N_G2{1};

% Define the folders Group 1
for i = 1:N_G1{1}
    functdir1{i} = fullfile(RootPath, Group{1}, myG1{i});
end

% Define the folders Group 2
for i = 1 : N_G2{1}
    functdir2{i} = fullfile(RootPath, Group{2}, myG2{i});
end
%% 1.1 Loading the data files
% This takes very long and needs a lot of RAM. On a "normal" computer you
% can load around 20/20 subjects, but if you want more, better run it on a
% server. 
mask ={};
brain_info ={};
n_dataset = 0;

% Data: cell array, each cell of size n_TP x n_masked_voxels
% Mask: n_voxels x 1 logical vector
% Header: the header (obtained by spm_vol) of one NIFTI file with proper
% data dimension and .mat information
[mask, brain_info] = createMask_CAP_SW(functdir1, prefix, selectMask);
Tstart = clock;

% Data Group 1: cell array, each cell of size n_TP x n_masked_voxels
TC ={};
FD ={};
[TC,brain,FD] = loadData_SW(functdir1, prefix, mask, brain_info);
FD1 = FD{:,:};
TC1 = TC{:,:};

% Data Group 2: cell array, each cell of size n_TP x n_masked_voxels
TC ={};
FD={};
[TC,brain,FD] = loadData_SW(functdir2, prefix, mask, brain_info);
FD2 = FD{:,:};
TC2 = TC{:,:};

% Seed: a n_masked_voxels x n_seed logical vector with seed information
disp(' ');
disp('-----------------------------------------------------');
disp('Load seed...');
[Seed] = loadseed_SW(functdir1, prefix, mask, brain_info, seedName,MasksPath);
disp('Seed successfully loaded!');

% Computes seed maps for each subject and for the population, using the
% data from the chosen reference population
TC = [TC1, TC2];
[~,AvgSeedMap] = CAP_Compute_SeedMap(TC1,Seed,1);

%% 2. Specifying the main parameters

% Threshold above which to select frames
T = 1; %more restricted = 0.84, CDF-1(0.80) = 0.84)

% Selection mode ('Threshold' or 'Percentage')
SelMode = 'Threshold';

% Threshold of FD above which to scrub out the frame and also the t-1 and
% t+1 frames (if you want another scrubbing setting, directly edit the
% code)
Tmot = 0.5;

% Type of used seed information: select between 'Average' or 'Union' ->
% Union only to be used if > 1 seed, quite difficult to interpret though. 
% I'd recommend - if you want to use two seeds - to do two individual CAP
% analyses with the two individual seeds

SeedType = 'Average'; 

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
[Xon1,p1,Indices1,idx_sep_seeds1,Xonp_scrub1] = CAP_find_activity(TC1,Seed,T,FD1,Tmot,SelMode,SeedType,SignMatrix);
[Xon2,p2,Indices2,idx_sep_seeds2,Xonp_scrub2] = CAP_find_activity(TC2,Seed,T,FD2,Tmot,SelMode,SeedType,SignMatrix);
    
% Percentage of retained frames across subjects
RetainedPercentage{1} = p1(3,:);
RetainedPercentage{2} = p2(3,:);

% Indices of the frames that have been retained (used later for metrics
% computations)
FrameIndices{1} = Indices1;
FrameIndices{2} = Indices2;

tmp_toplot1 = ConcatMat(RetainedPercentage(1),1,1,N_G1,'FD');
tmp_toplot2 = ConcatMat(RetainedPercentage(2),1,1,N_G2,'FD');

tmp_toplot = zeros(2,N_G2{1});
tmp_toplot(1,1:N_G1{1}) = tmp_toplot1;
tmp_toplot(2,1:N_G2{1}) = tmp_toplot2;
tmp_toplot(tmp_toplot == 0) = NaN;

% Displays the violin plot of subject scrubbing percentage for the
% reference population
TPViolin = figure;
axes1 = axes('Parent',TPViolin);

% Colors used in plotting of all populations
PopColor{1} = [255,255,180; 219,224,252; 188,252,188; 230,230,230]/255;
PopColor{2} = [130,48,48; 51,75,163; 59,113,86; 0, 0, 0]/255;

%Create plot
[~,~,TPViolin] = MakeViolin_SW(tmp_toplot,axes1,{'Group 1' 'Group 2'},'Frames ret. [%]',PopColor,1,2);
set(TPViolin,'Visible','on');

%Save plot as .jpg and .fig 
saveas(gcf,fullfile(SavePath,fancyName,'RetainedFrames.jpg'));
saveas(gcf,fullfile(SavePath,fancyName,'RetainedFrames.fig'));
disp('-----------------------------------------------------');
disp(' ');
disp(['Figure Frames retained has been saved in ' SavePath]);
close;
%% 4. Consensus clustering (if wished to determine the optimum K)

% The input to PCA should have a dimensionality n_dimensions x
% n_datapoints; since we consider a population of subjects in a cell array
% (each cell with size n_dims x n_samples_persubj), we want to concatenate
% these cells into one giant data matrix, which we feed to the PCA
% function. I will denote the number of dimensions (or voxels) by V, and
% the number of total time points across subjects by T.

% Change dimension of Xon1 to feed into PCA

% This is an add-on to the original TbCAPs toolbox by Bolton et al,
% 2020. Steps are explained below. 

Xon1_pca = [];

for i = 1:N_G1{1,1}

    Xon1_pca = [Xon1_pca, Xon1{1,i}];
    
end

[U, W,Eigenvals,mu] = ComputePCA_SW(Xon1_pca);

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
K_range = 2:20;

% Have each of these run in a separate process on the server =)
disp('Running Consensus Clustering...')

%Consensus clustering using PCA reduced dataset
[Consensus] = CAP_ConsensusClustering({W},K_range,'items',Pcc/100,N,'correlation');

%Plot Consensus Matrix
ConsensusMatrixPlot_SW(Consensus,SavePath,fancyName, K_range)

% Calculates the quality metrics
[CDF,PAC] = ComputeClusteringQuality(Consensus,K_range);

% % Calculates the quality metrics
% [~,Qual] = ComputeClusteringQuality(Consensus,[]);
%save(fullfile(SavePath,fancyName,['ConsensusClustering_Range' num2str(K_range(1)) '_to_' num2str(K_range(end)) '_' fancyName '.mat']))
disp(' ');
disp('Consensus Cluster successfully performed!');

Ttotal=etime(clock, Tstart);
disp(['** Consensus clustering completed. Total time: ' num2str(Ttotal/60,'%3.1f') ' min.']);

% Qual should be inspected to determine the best cluster number(s)
CCPlot = figure;
ax1 = axes(CCPlot);

%set(CCPlot,'Visible','on');
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
% Check your stability plot and your consensus matrices. The stability of a
% cluster must be as close to 1 as possible and as "flat" as possible.
% Moreover, the boarders of the consensus matrices must be as crisp as
% possible. A good cluster therefore is represented by crisp boarders in
% the corresponding consensus matrix plot and by a high (close to 1)
% stability. 
% K_opt = 2 is trivial, so don't use that. 

%K_opt = 3;
prompt = 'What is the optimal cluster size? ';
K_opt = input(prompt);
close;

%% 5. Clustering into CAPs

[CAP,Disp,Std_Clusters,idx1,CorrDist,sfrac] = Run_Clustering_PCA(W,...
    K_opt,mask{1},brain_info{1},Pp,Pn,n_rep,idx_sep_seeds1,SeedType);
 
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
    % only makes sense, if you have more than one seed and use union

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
    
%% 6. Assignment of the frames from population 2

% Parameter that governs the stringency of assignment: if Ap = 5%, we
% assign a frame to a CAP if spatial correlation exceeds the 5th percentile
% of the distribution of spatial correlations between the CAP, and its
% constituting frames
Ap = 5;

idx2 = CAP_AssignFrames(CAP,cell2mat(Xon2),CorrDist,Ap)';


%% 7. Computing metrics

[ExpressionMap1,Counts1,Entries1,Avg_Duration1,Duration1,TransitionProbabilities1,...
    From_Baseline1,To_Baseline1,Baseline_resilience1,Resilience1,Betweenness1,...
    InDegree1,OutDegree1,SubjectEntries1] = Compute_Metrics_simpler(idx1,...
    Indices1.kept.active,Indices1.scrubbedandactive,K_opt,TR);

[ExpressionMap2,Counts2,Entries2,Avg_Duration2,Duration2,TransitionProbabilities2,...
    From_Baseline2,To_Baseline2,Baseline_resilience2,Resilience2,Betweenness2,...
    InDegree2,OutDegree2,SubjectEntries2] = Compute_Metrics_simpler(idx2,...
    Indices2.kept.active,Indices2.scrubbedandactive,K_opt,TR);

% Plot metrics Transition matrix for Group 1

tmp_toplot = squeeze(mean(TransitionProbabilities1,3));
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

saveas(gcf,fullfile(SavePath,fancyName,'TransitionProbabilityMatrix_Group1.jpg'));
saveas(gcf,fullfile(SavePath,fancyName,'TransitionProbabilityMatrix_Group1.fig'));
close;

% For Group 2
tmp_toplot = squeeze(mean(TransitionProbabilities2,3));
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

saveas(gcf,fullfile(SavePath,fancyName,'TransitionProbabilityMatrix_Group2.jpg'));
saveas(gcf,fullfile(SavePath,fancyName,'TransitionProbabilityMatrix_Group2.fig'));
close;

disp(' ');
disp('-----------------------------------------------------');
disp(['Figure Transition Probability Matrix has been saved in ' SavePath]);
disp(' ');

% Dynamic state plotting

% Makes the graph visible
% Concatenates information from the different datasets
tmp_toplot = [];
ExpressionMap{1}=ExpressionMap1;
ExpressionMap{2}=ExpressionMap2;
for i = 1:n_datasets
    tmp_toplot = [tmp_toplot; ExpressionMap{i}; 0*ones(5,size(TC{1},1))];
end
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
    for i = 1:n_datasets
        % Cumulative distribution for the state that we want to be displayed (i.e.
        % the state from the popup menu)
        %handles.TPMCum{i} = cumsum(TPM{i} == 1; % 1 = cap --> REPLACE LATER WITH CAP, TPM = ExpressionIndices
        if i == 1
            TPMCum{i} = cumsum(ExpressionMap1 == s,2); % 1 = cap --> REPLACE LATER WITH CAP, TPM = ExpressionIndices
        elseif i == 2
            TPMCum{i} = cumsum(ExpressionMap2 == s,2);
        end
        % Average of the considered state across subjects
        tmp_TPMCum{i} = mean(TPMCum{i},1);
    end
    clear i
    % Similarly as above, we plot time if we have a valid TR; else, we plot
    subplot(size(CAP,1),1,s)
        for i = 1:n_datasets
            for j = 1:size(TPMCum{i},1)
                plot(((1:size(ExpressionMap1,2))-1)*TR,...
                    TPMCum{i}(j,:),...
                    'Color',PopColor{1}(i,:));
                hold('on');
            end
        end

    for i = 1:n_datasets
       plot(((1:size(ExpressionMap1,2))-1)*TR,...
            tmp_TPMCum{i},...
            'LineWidth',2,'Color',PopColor{2}(i,:));
        xlabel('Time [s]','FontSize',10);
        xlim([0,(size(ExpressionMap1,2)-1)*TR]);
    end
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

Outputs.SpatioTemporalSelection.RetainedFramesPerSeed{1} = idx_sep_seeds1;
Outputs.SpatioTemporalSelection.RetainedFramesPerSeed{2} = idx_sep_seeds2;
Outputs.SpatioTemporalSelection.Indices{1}=Indices1;
Outputs.SpatioTemporalSelection.Indices{1}=Indices2;
Outputs.SpatioTemporalSelection.PercentageRetainedFrames = RetainedPercentage;
Outputs.SpatioTemporalSelection.AverageCorrelationMap = AvgSeedMap;
Outputs.KMeansClustering.ConsensusQuality = PAC; 
Outputs.KMeansClustering.CoActivationPatternsDispersion = Disp;
Outputs.KMeansClustering.CoActivationPatterns = CAP;
Outputs.KMeansClustering.CoActivationPatternsZScored = CAP_Zscore(CAP);
Outputs.KMeansClustering.CoActivationPatternsSTD = Std_Clusters; 
Outputs.KMeansClustering.AssignmentsToCAPs{1} = idx1;
Outputs.KMeansClustering.AssignmentsToCAPs{2} = idx2;
Outputs.Metrics.CAPExpressionIndices{1} = ExpressionMap1;
Outputs.Metrics.CAPExpressionIndices{2} = ExpressionMap2;
Outputs.Metrics.Occurrences{1} = Counts1;
Outputs.Metrics.Occurrences{2} = Counts2;
Outputs.Metrics.NumberEntries{1} = Entries1;
Outputs.Metrics.NumberEntries{2} = Entries2;
Outputs.Metrics.AverageExpressionDuration{1} = Avg_Duration1;
Outputs.Metrics.AverageExpressionDuration{2} = Avg_Duration2;
Outputs.Metrics.AllExpressionDurations{1} = Duration1;
Outputs.Metrics.AllExpressionDurations{2} = Duration2;
Outputs.Metrics.TransitionProbabilities{1} = TransitionProbabilities1;
Outputs.Metrics.TransitionProbabilities{2} = TransitionProbabilities2;
Outputs.Metrics.FractionCAPFramesPerSeedCombination = sfrac;
Outputs.Metrics.CAPEntriesFromBaseline{1} = From_Baseline1;
Outputs.Metrics.CAPEntriesFromBaseline{2} = From_Baseline2;
Outputs.Metrics.CAPExitsToBaseline{1} = To_Baseline1;
Outputs.Metrics.CAPExitsToBaseline{2} = To_Baseline2;
Outputs.Metrics.CAPResilience{1} = Resilience1;
Outputs.Metrics.CAPResilience{2} = Resilience2;
Outputs.Metrics.BaselineResilience{1} = Baseline_resilience1;
Outputs.Metrics.BaselineResilience{2} = Baseline_resilience2;
Outputs.Metrics.BetweennessCentrality{1} = Betweenness1;
Outputs.Metrics.BetweennessCentrality{2} = Betweenness2;
Outputs.Metrics.CAPInDegree{1} = InDegree1;
Outputs.Metrics.CAPInDegree{2} = InDegree2;
Outputs.Metrics.CAPOutDegree{1} = OutDegree1;
Outputs.Metrics.CAPOutDegree{2} = OutDegree2;
Outputs.Metrics.SubjectCounts{1} = SubjectEntries1;
Outputs.Metrics.SubjectCounts{2} = SubjectEntries2;

%%Uncomment next three lines, if you want to save heavy outputs
%HeavyOutputs.SpatioTemporalSelection.ClusteredFrames{1} = Xon1;
%HeavyOutputs.SpatioTemporalSelection.ClusteredFrames{2} = Xon2;
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

% Write temporal metrics as excel file for further statistical analyses
CAP_writeExcelMetrics_TwoPop_SW(SavePath,fancyName,Outputs,nTP,mySubj,N_Subj{1})
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