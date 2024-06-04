%% This is the main script containing the routines necessary for the use
% of the co-activation pattern analysis (CAP) visualization toolbox
%
% Original scripts implemented and written by Thomas A. W. Bolton, from
% the Medical Image Processing Laboratory (MIP:Lab), EPFL, Switzerland
% 
% Script has been adapted by Samantha Weber, from Aybek FND Lab, UniBe,
% Switzerland
%
% For concerns, please write to samantha.weber@extern.insel.ch
%
% Version 1.0 (March 2020): first usable version of the toolbox, and
% start of use by Samantha Weber for visualization of CAPs after running
% PCA CAPs analysis script-based. 

% Use Script_twopop_SW.m or Script_onepop_SW.m before using this toolbox.


function varargout = CAP_SW(varargin)
%CAP_SW_V1 MATLAB code file for CAP_SW.fig
%      CAP_SW, by itself, creates a new CAP_SW or raises the existing
%      singleton*.
%
%      H = CAP_SW returns the handle to a new CAP_SW or the handle to
%      the existing singleton*.
%
%      CAP_SW('Property','Value',...) creates a new CAP_SW using the
%      given property value pairs. Unrecognized properties are passed via
%      varargin to CAP_SW_v1_OpeningFcn.  This calling syntax produces a
%      warning when there is an existing singleton*.
%
%      CAP_SW('CALLBACK') and CAP_SW('CALLBACK',hObject,...) call the
%      local function named CALLBACK in CAP_SW.M with the given input
%      arguments.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help CAP_SW_v1

% Last Modified by GUIDE v2.5 06-Apr-2022 12:36:01



% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CAP_SW_OpeningFcn, ...
                   'gui_OutputFcn',  @CAP_SW_OutputFcn, ...
                   'gui_LayoutFcn',  [], ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
   gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before CAP_SW is made visible.
function CAP_SW_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.

% Sets warnings off
warning('off');

% Choose default command line output for CAP_SW
handles.output = hObject;

clc; 

%%%% Data Loading %%%%

% Log containing the different events summoned from the toolbox
handles.Log = {};

% Colors used in plotting of all populations
handles.PopColor{1} = [255,255,180; 219,224,252; 188,252,188; 230,230,230]/255;
handles.PopColor{2} = [130,48,48; 51,75,163; 59,113,86; 0, 0, 0]/255;

% Update handles structure
guidata(hObject, handles);

% --- Outputs from this function are returned to the command line.
function varargout = CAP_SW_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes during object creation, after setting all properties.
function TitleText_CreateFcn(hObject, eventdata, handles)
% hObject    handle to TitleText (see GCBO)

% --- Executes during object deletion, before destroying properties.
function TitleText_DeleteFcn(hObject, eventdata, handles)
% hObject    handle to TitleText (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% SECTION 1: LOADING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% --- Executes on button press in Load_Button.
function varargout = Load_Button_Callback(hObject, eventdata, handles)

handles = ClearSection(eventdata,handles);

[file, path]=uigetfile('*.mat',...
    'Select PCA CAPs output file');
disp('... loading Data')
disp(' ')
load(fullfile(path,file),'Outputs','Parameters','brain');
disp('... Data loading complete!')

% asign previous outputs to base
handles.Outputs = Outputs;
handles.Parameters = Parameters;
assignin('base','Outputs', Outputs);
assignin('base','Parameters',Parameters);
assignin('base','brain', brain);

%load previous mask
mask = handles.Parameters.Inputs.Mask;
assignin('base','mask',mask);
%load dimensions brain
brain_info = handles.Parameters.Inputs.DataHeader;
assignin('base','brain_info',brain_info);
%load number of clusters
Kmax = handles.Parameters.KMeansClustering.NumberClusters;
assignin('base','Kmax',Kmax);
%Load number of seeds
n_seed = handles.Parameters.SpatioTemporalSelection.NumberSeeds;
assignin('base','n_seed',n_seed)
%load seed type
SeedType = handles.Parameters.SpatioTemporalSelection.SeedType;
assignin('base','SeedType',SeedType)
%load seeds
seed = handles.Parameters.Inputs.Seeds;
assignin('base','seed',seed);
%Load CAPs & STDCAP
CAP = handles.Outputs.KMeansClustering.CoActivationPatterns;%ZScored;
%CAP = handles.Outputs.KMeansClustering.CoActivationPatternsZScored;
assignin('base','CAP',CAP)
STDCAP = handles.Outputs.KMeansClustering.CoActivationPatternsSTD;
assignin('base','STDCAP',STDCAP)
%load fraction CAP frames per seed combination
sfrac = handles.Outputs.Metrics.FractionCAPFramesPerSeedCombination;
assignin('base','sfrac',sfrac)
%load dispersion
Disp = handles.Outputs.KMeansClustering.CoActivationPatternsDispersion;
assignin('base','Disp',Disp)

%Assign them to handles
handles.brain = brain;
handles.Kmax = Kmax;
handles.mask = mask;
handles.brain_info = brain_info;
handles.n_seed = n_seed;
handles.SeedType = SeedType;
handles.seed = seed;
handles.CAP = CAP;
handles.STDCAP = STDCAP;
handles.sfrac = sfrac;
handles.Disp = Disp;

% % We can now enable the seed selection
set(handles.ShowCAPsButton,'Enable','on');

guidata(hObject, handles);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% SECTION 2: SHOW CAPS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Upon clicking on 'Show CAPs'
function ShowCAPsButton_Callback(hObject, eventdata, handles)
handles.CAP = handles.Outputs.KMeansClustering.CoActivationPatterns;

set(handles.text_similarity,'Visible','on');
set(handles.Zscore,'Visible','on');
set(handles.PIE_S1,'Visible','off');
set(handles.PIE_S2,'Visible','off');
set(handles.PIE_S3,'Visible','off');
set(handles.PIE_S1S2,'Visible','off');
set(handles.PIE_S2S3,'Visible','off');
set(handles.PIE_S1S3,'Visible','off');
set(handles.PIE_S1S2S3,'Visible','off');

% Resets the CAP parameters (CAPs, standard deviation within CAPs and
% indices of the CAPs to which all retained frames were assigned)
% Resets the graph display of the CAP colorbar
handles = ResetGraphDisplay(handles.ColorbarCAP,handles);

% Reset all graph displays for the CAPs
tmpX = {handles.CAP1X,handles.CAP2X,handles.CAP3X,handles.CAP4X,handles.CAP5X};
tmpY = {handles.CAP1Y,handles.CAP2Y,handles.CAP3Y,handles.CAP4Y,handles.CAP5Y};
tmpZ = {handles.CAP1Z,handles.CAP2Z,handles.CAP3Z,handles.CAP4Z,handles.CAP5Z};
tmpF = {handles.CAP1_Frames,handles.CAP2_Frames,handles.CAP3_Frames,handles.CAP4_Frames,handles.CAP5_Frames};

for i_CAP = 1:5
    set(tmpF{i_CAP},'Visible','off');
    handles = ResetGraphDisplay(tmpX{i_CAP},handles);
    handles = ResetGraphDisplay(tmpY{i_CAP},handles);
    handles = ResetGraphDisplay(tmpZ{i_CAP},handles);
end

% Resets the sliders and the textboxes for the CAPs
set(handles.CAP_SliderX,'Visible','off');
set(handles.CAP_SliderY,'Visible','off');
set(handles.CAP_SliderZ,'Visible','off');
set(handles.CAP_XC,'Visible','off');
set(handles.CAP_YC,'Visible','off');
set(handles.CAP_ZC,'Visible','off');

% Resets the slider and textbox for the CAPs visualization threshold
set(handles.TVIS_Slider,'Visible','off');
set(handles.TVIS,'Visible','off');

% Resets the pie charts
handles = ResetGraphDisplay(handles.pie1,handles);
handles = ResetGraphDisplay(handles.pie2,handles);
handles = ResetGraphDisplay(handles.pie3,handles);
handles = ResetGraphDisplay(handles.pie4,handles);
handles = ResetGraphDisplay(handles.pie5,handles);

handles = ResetGraphDisplay(handles.ColorbarSimMat,handles);
handles = ResetGraphDisplay(handles.CAP_Mat,handles);

% handles = ClearSection4(eventdata,handles);

if size(handles.Outputs.KMeansClustering.AssignmentsToCAPs,2) == 1
    handles.idx = handles.Outputs.KMeansClustering.AssignmentsToCAPs;
else 
    handles.idx = handles.Outputs.KMeansClustering.AssignmentsToCAPs{1};
end

% Makes the sliders visible, and the related text too (CAP MNI coordinates)
set(handles.CAP_SliderX,'Visible','on');
set(handles.CAP_SliderY,'Visible','on');
set(handles.CAP_SliderZ,'Visible','on');
set(handles.CAP_XC,'Visible','on');
set(handles.CAP_YC,'Visible','on');
set(handles.CAP_ZC,'Visible','on');
set(handles.CAP_XC,'String',['X: ',sprintf('%.2f',get(handles.CAP_SliderX,'Value'))]);
set(handles.CAP_YC,'String',['Y: ',sprintf('%.2f',get(handles.CAP_SliderY,'Value'))]);
set(handles.CAP_ZC,'String',['Z: ',sprintf('%.2f',get(handles.CAP_SliderZ,'Value'))]);

% Computation of the similarity
SimMat = corr(handles.CAP',handles.CAP');
SimMat(isnan(SimMat))=0;

% Graph set visible, and plotting
handles = ResetGraphDisplay(handles.CAP_Mat,handles);
set(handles.CAP_Mat,'Visible','on');
imagesc(SimMat,'Parent',handles.CAP_Mat);

tmp_cb2 = cbrewer('div','RdBu',1000);
tmp_cb2(tmp_cb2 < 0) = 0;

colormap(handles.CAP_Mat,flipud(tmp_cb2));

% Correlation ranges from -1 to 1, so this is what we make the graph
% colorbar vary within. We also make the graph square and remove the axes
caxis(handles.CAP_Mat,[-1 1]);
axis(handles.CAP_Mat,'square','on');
axis(handles.CAP_Mat,'off');

% Addition of the colorbar just below
set(handles.ColorbarSimMat,'Visible','on');
handles.ColorbarSimMat = Create_CAP_colorbar(-1,1,0.5,0,'',...
    handles.ColorbarSimMat,'Vertical','div','RdBu',1000);

% If using the 'Union' option...

if strcmp(handles.SeedType,'Union')
    
    % Custom colormap
    custom_cm = 1/255*[211,36,36;11,170,65;51,75,163;255,255,180;186,59,204;58,221,221;242,242,242];
    
    % Graph displays are stored in a common tmp_sfrac cell array
    tmp_sfrac = {handles.pie1,handles.pie2,handles.pie3,handles.pie4,...
        handles.pie5};
    
    % The pie charts for each cluster are created
    for cc = 1:handles.Kmax
        
        % Pie charts
        set(tmp_sfrac{cc},'Visible','on');
        for tt = 1:size(handles.sfrac,3)
            lab{tt} = '';
        end
        
        pie(tmp_sfrac{cc},realmin*ones(size(handles.sfrac,3),1)+squeeze(mean(handles.sfrac(:,cc,:),1)),lab);
        caxis(tmp_sfrac{cc},[1,7]);
        
        switch handles.n_seed
            case 1
                errordlg('You managed the impossible, congratulations!');
            case 2
                colormap(tmp_sfrac{cc},(custom_cm));
                set(handles.PIE_S1,'Visible','on');
                set(handles.PIE_S2,'Visible','on');
                set(handles.PIE_S1S2,'Visible','on');
                set(handles.PIE_S1S2,'ForeGroundColor',[51,75,163]/255);
            case 3
                colormap(tmp_sfrac{cc},(custom_cm));
                set(handles.PIE_S1,'Visible','on');
                set(handles.PIE_S2,'Visible','on');
                set(handles.PIE_S3,'Visible','on');
                set(handles.PIE_S2S3,'Visible','on');
                set(handles.PIE_S1S2,'Visible','on');
                set(handles.PIE_S1S2,'ForeGroundColor',[243,243,139]/255);
                set(handles.PIE_S1S3,'Visible','on');
                set(handles.PIE_S1S2S3,'Visible','on');
        end
    end
end

% Same for the slider for the visualization threshold
set(handles.TVIS,'Visible','on');
set(handles.TVIS_Slider,'Visible','on'); 
set(handles.TVIS,'String',['Tv: ',sprintf('%.2f',get(handles.TVIS_Slider,'Value'))]);

% Makes the colorbar for the CAPs visible
handles.ColorbarCAP = Create_CAP_colorbar(-1.5,1.5,0.5,get(handles.TVIS_Slider,'Value'),'',handles.ColorbarCAP,'Horizontal','div','RdBu',1000);
set(handles.ColorbarCAP,'Visible','on');

% Concatenates all CAP information into metavariables for easier subsequent
% changes
tmpX = {handles.CAP1X,handles.CAP2X,handles.CAP3X,handles.CAP4X,handles.CAP5X};
tmpY = {handles.CAP1Y,handles.CAP2Y,handles.CAP3Y,handles.CAP4Y,handles.CAP5Y};
tmpZ = {handles.CAP1Z,handles.CAP2Z,handles.CAP3Z,handles.CAP4Z,handles.CAP5Z};
tmpF = {handles.CAP1_Frames,handles.CAP2_Frames,handles.CAP3_Frames,handles.CAP4_Frames,handles.CAP5_Frames};

%handles.Kmax = handles.Parameters.KMeansClustering.NumberClusters;
% For each CAP...
for i_CAP = 1:handles.Kmax
    
    % Clears the display for each dimension
    cla(tmpX{i_CAP});
    cla(tmpY{i_CAP});
    cla(tmpZ{i_CAP});
    
    % Plots the new slice for each dimension
    tmpX{i_CAP} = plot_slice(handles.CAP(i_CAP,:),...
        get(handles.TVIS_Slider,'Value'),1.5,handles.mask{1},handles.brain,handles.brain_info{1},...
        'X',get(handles.CAP_SliderX,'Value'),tmpX{i_CAP});

    tmpY{i_CAP} = plot_slice(handles.CAP(i_CAP,:),...
        get(handles.TVIS_Slider,'Value'),1.5,handles.mask{1},handles.brain,handles.brain_info{1},...
        'Y',get(handles.CAP_SliderY,'Value'),tmpY{i_CAP});
    
    tmpZ{i_CAP} = plot_slice(handles.CAP(i_CAP,:),...
        get(handles.TVIS_Slider,'Value'),1.5,handles.mask{1},handles.brain,handles.brain_info{1},...
        'Z',get(handles.CAP_SliderZ,'Value'),tmpZ{i_CAP});

    % Sets the frame percentage text visible and at the right value (number
    % of frames from a CAP/total frame number, and then percentage that it
    % stands for)
    set(tmpF{i_CAP},'Visible','on');
    set(tmpF{i_CAP},'String',{[num2str(sum(handles.idx(:)==i_CAP)),'/',...
        num2str(size(handles.idx,1))],[sprintf('%.2f',...
        sum(handles.idx(:)==i_CAP)/size(handles.idx(:),1)*100),' [%]']});
end

guidata(hObject, handles);

%% Sliders for CAP visualization (MNI coordinates)
% When changing along a slider, we want to update the graphs and the text of
% the MNI coordinate below the slider
function CAP_SliderX_Callback(hObject, eventdata, handles)

set(handles.CAP_XC,'String',['X: ',sprintf('%.2f',get(hObject,'Value'))]);
tmp_struct = {handles.CAP1X,handles.CAP2X,handles.CAP3X,handles.CAP4X,handles.CAP5X};

for i_CAP = 1:handles.Kmax
    cla(tmp_struct{i_CAP});  
    tmp_struct{i_CAP} = plot_slice(handles.CAP(i_CAP,:),get(handles.TVIS_Slider,'Value'),...
        1.5,handles.mask{1},handles.brain,...
        handles.brain_info{1},'X',get(hObject,'Value'),tmp_struct{i_CAP});
end

guidata(hObject, handles); 

function CAP_SliderX_CreateFcn(hObject, eventdata, handles)

if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

guidata(hObject,handles);

% Y dimension slider
function CAP_SliderY_Callback(hObject, eventdata, handles)

set(handles.CAP_YC,'String',['Y: ',sprintf('%.2f',get(hObject,'Value'))]);
tmp_struct = {handles.CAP1Y,handles.CAP2Y,handles.CAP3Y,handles.CAP4Y,handles.CAP5Y};

for i_CAP = 1:handles.Kmax
    cla(tmp_struct{i_CAP});
    tmp_struct{i_CAP} = plot_slice(handles.CAP(i_CAP,:),get(handles.TVIS_Slider,'Value'),...
        1.5,handles.mask{1},handles.brain,...
        handles.brain_info{1},'Y',get(hObject,'Value'),tmp_struct{i_CAP});
end

guidata(hObject,handles);

function CAP_SliderY_CreateFcn(hObject, eventdata, handles)

if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

guidata(hObject,handles);

% Z dimension slider
function CAP_SliderZ_Callback(hObject, eventdata, handles)

set(handles.CAP_ZC,'String',['Z: ',sprintf('%.2f',get(hObject,'Value'))]);
tmp_struct = {handles.CAP1Z,handles.CAP2Z,handles.CAP3Z,handles.CAP4Z,handles.CAP5Z};

for i_CAP = 1:handles.Kmax
   
    cla(tmp_struct{i_CAP});
    tmp_struct{i_CAP} = plot_slice(handles.CAP(i_CAP,:),get(handles.TVIS_Slider,'Value'),...
        1.5,handles.mask{1},handles.brain,handles.brain_info{1},'Z',get(hObject,'Value'),tmp_struct{i_CAP});
end

guidata(hObject,handles);

function CAP_SliderZ_CreateFcn(hObject, eventdata, handles)

if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

guidata(hObject,handles);

%% Sliders for threshold visualization (CAP analysis)
% Again, we want to update the slices and the text if we change those
% sliders
function TVIS_Slider_Callback(hObject, eventdata, handles)

% The text is changed
set(handles.TVIS,'String',['Tv: ',sprintf('%.2f',get(hObject,'Value'))]);

% The colorbar graph is modified to suit the new threshold value
cla(handles.ColorbarCAP);
handles.ColorbarCAP = Create_CAP_colorbar(-1.5,1.5,0.5,get(hObject,'Value'),'',handles.ColorbarCAP,'Horizontal','div','RdBu',1000);

% The brain slices are replotted
tmpX = {handles.CAP1X,handles.CAP2X,handles.CAP3X,handles.CAP4X,handles.CAP5X};
tmpY = {handles.CAP1Y,handles.CAP2Y,handles.CAP3Y,handles.CAP4Y,handles.CAP5Y};
tmpZ = {handles.CAP1Z,handles.CAP2Z,handles.CAP3Z,handles.CAP4Z,handles.CAP5Z};

for i_CAP = 1:handles.Kmax
    
    cla(tmpX{i_CAP});
    cla(tmpY{i_CAP});
    cla(tmpZ{i_CAP});
    
    tmpX{i_CAP} = plot_slice(handles.CAP(i_CAP,:),get(hObject,'Value'),1.5,...
        handles.mask{1},handles.brain,...
        handles.brain_info{1},'X',get(handles.CAP_SliderX,'Value'),tmpX{i_CAP});
    
    tmpY{i_CAP} = plot_slice(handles.CAP(i_CAP,:),get(hObject,'Value'),1.5,...
        handles.mask{1},handles.brain,...
        handles.brain_info{1},'Y',get(handles.CAP_SliderY,'Value'),tmpY{i_CAP});
    
    tmpZ{i_CAP} = plot_slice(handles.CAP(i_CAP,:),get(hObject,'Value'),1.5,...
        handles.mask{1},handles.brain,...
        handles.brain_info{1},'Z',get(handles.CAP_SliderZ,'Value'),tmpZ{i_CAP});
end

guidata(hObject,handles);

function TVIS_Slider_CreateFcn(hObject, eventdata, handles)

if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% Executes on button press in Zscore.
function Zscore_Callback(hObject, eventdata, handles)

handles.CAP = handles.Outputs.KMeansClustering.CoActivationPatternsZScored;
set(handles.text_similarity,'Visible','on');
set(handles.Zscore,'Visible','on');
set(handles.PIE_S1,'Visible','off');
set(handles.PIE_S2,'Visible','off');
set(handles.PIE_S3,'Visible','off');
set(handles.PIE_S1S2,'Visible','off');
set(handles.PIE_S2S3,'Visible','off');
set(handles.PIE_S1S3,'Visible','off');
set(handles.PIE_S1S2S3,'Visible','off');

% Resets the CAP parameters (CAPs, standard deviation within CAPs and
% indices of the CAPs to which all retained frames were assigned)
% Resets the graph display of the CAP colorbar
handles = ResetGraphDisplay(handles.ColorbarCAP,handles);

% Reset all graph displays for the CAPs
tmpX = {handles.CAP1X,handles.CAP2X,handles.CAP3X,handles.CAP4X,handles.CAP5X};
tmpY = {handles.CAP1Y,handles.CAP2Y,handles.CAP3Y,handles.CAP4Y,handles.CAP5Y};
tmpZ = {handles.CAP1Z,handles.CAP2Z,handles.CAP3Z,handles.CAP4Z,handles.CAP5Z};
tmpF = {handles.CAP1_Frames,handles.CAP2_Frames,handles.CAP3_Frames,handles.CAP4_Frames,handles.CAP5_Frames};

for i_CAP = 1:5
    set(tmpF{i_CAP},'Visible','off');
    handles = ResetGraphDisplay(tmpX{i_CAP},handles);
    handles = ResetGraphDisplay(tmpY{i_CAP},handles);
    handles = ResetGraphDisplay(tmpZ{i_CAP},handles);
end

% Resets the sliders and the textboxes for the CAPs
set(handles.CAP_SliderX,'Visible','off');
set(handles.CAP_SliderY,'Visible','off');
set(handles.CAP_SliderZ,'Visible','off');
set(handles.CAP_XC,'Visible','off');
set(handles.CAP_YC,'Visible','off');
set(handles.CAP_ZC,'Visible','off');

% Resets the slider and textbox for the CAPs visualization threshold
set(handles.TVIS_Slider,'Visible','off');
set(handles.TVIS,'Visible','off');

% Resets the pie charts
handles = ResetGraphDisplay(handles.pie1,handles);
handles = ResetGraphDisplay(handles.pie2,handles);
handles = ResetGraphDisplay(handles.pie3,handles);
handles = ResetGraphDisplay(handles.pie4,handles);
handles = ResetGraphDisplay(handles.pie5,handles);

handles = ResetGraphDisplay(handles.ColorbarSimMat,handles);
handles = ResetGraphDisplay(handles.CAP_Mat,handles);

% handles = ClearSection4(eventdata,handles);

if size(handles.Outputs.KMeansClustering.AssignmentsToCAPs,2) == 1
    handles.idx = handles.Outputs.KMeansClustering.AssignmentsToCAPs;
else 
    handles.idx = handles.Outputs.KMeansClustering.AssignmentsToCAPs{1};
end

% Makes the sliders visible, and the related text too (CAP MNI coordinates)
set(handles.CAP_SliderX,'Visible','on');
set(handles.CAP_SliderY,'Visible','on');
set(handles.CAP_SliderZ,'Visible','on');
set(handles.CAP_XC,'Visible','on');
set(handles.CAP_YC,'Visible','on');
set(handles.CAP_ZC,'Visible','on');
set(handles.CAP_XC,'String',['X: ',sprintf('%.2f',get(handles.CAP_SliderX,'Value'))]);
set(handles.CAP_YC,'String',['Y: ',sprintf('%.2f',get(handles.CAP_SliderY,'Value'))]);
set(handles.CAP_ZC,'String',['Z: ',sprintf('%.2f',get(handles.CAP_SliderZ,'Value'))]);

% Computation of the similarity
SimMat = corr(handles.CAP',handles.CAP');
SimMat(isnan(SimMat))=0;

% Graph set visible, and plotting
handles = ResetGraphDisplay(handles.CAP_Mat,handles);
set(handles.CAP_Mat,'Visible','on');
imagesc(SimMat,'Parent',handles.CAP_Mat);

tmp_cb2 = cbrewer('div','RdBu',1000);
tmp_cb2(tmp_cb2 < 0) = 0;

colormap(handles.CAP_Mat,flipud(tmp_cb2));

% Correlation ranges from -1 to 1, so this is what we make the graph
% colorbar vary within. We also make the graph square and remove the axes
caxis(handles.CAP_Mat,[-1 1]);
axis(handles.CAP_Mat,'square','on');
axis(handles.CAP_Mat,'off');

% Addition of the colorbar just below
set(handles.ColorbarSimMat,'Visible','on');
handles.ColorbarSimMat = Create_CAP_colorbar(-1,1,0.5,0,'',...
    handles.ColorbarSimMat,'Vertical','div','RdBu',1000);

% If using the 'Union' option...

if strcmp(handles.SeedType,'Union')
    
    % Custom colormap
    custom_cm = 1/255*[211,36,36;11,170,65;51,75,163;255,255,180;186,59,204;58,221,221;242,242,242];
    
    % Graph displays are stored in a common tmp_sfrac cell array
    tmp_sfrac = {handles.pie1,handles.pie2,handles.pie3,handles.pie4,...
        handles.pie5};
    
    % The pie charts for each cluster are created
    for cc = 1:handles.Kmax
        
        % Pie charts
        set(tmp_sfrac{cc},'Visible','on');
        for tt = 1:size(handles.sfrac,3)
            lab{tt} = '';
        end
        
        pie(tmp_sfrac{cc},realmin*ones(size(handles.sfrac,3),1)+squeeze(mean(handles.sfrac(:,cc,:),1)),lab);
        caxis(tmp_sfrac{cc},[1,7]);
        
        switch handles.n_seed
            case 1
                errordlg('You managed the impossible, congratulations!');
            case 2
                colormap(tmp_sfrac{cc},(custom_cm));
                set(handles.PIE_S1,'Visible','on');
                set(handles.PIE_S2,'Visible','on');
                set(handles.PIE_S1S2,'Visible','on');
                set(handles.PIE_S1S2,'ForeGroundColor',[51,75,163]/255);
            case 3
                colormap(tmp_sfrac{cc},(custom_cm));
                set(handles.PIE_S1,'Visible','on');
                set(handles.PIE_S2,'Visible','on');
                set(handles.PIE_S3,'Visible','on');
                set(handles.PIE_S2S3,'Visible','on');
                set(handles.PIE_S1S2,'Visible','on');
                set(handles.PIE_S1S2,'ForeGroundColor',[243,243,139]/255);
                set(handles.PIE_S1S3,'Visible','on');
                set(handles.PIE_S1S2S3,'Visible','on');
        end
    end
end

% Same for the slider for the visualization threshold
set(handles.TVIS,'Visible','on');
set(handles.TVIS_Slider,'Visible','on'); 
set(handles.TVIS,'String',['Tv: ',sprintf('%.2f',get(handles.TVIS_Slider,'Value'))]);

% Makes the colorbar for the CAPs visible
handles.ColorbarCAP = Create_CAP_colorbar(-1.5,1.5,0.5,get(handles.TVIS_Slider,'Value'),'',handles.ColorbarCAP,'Horizontal','div','RdBu',1000);
set(handles.ColorbarCAP,'Visible','on');

% Concatenates all CAP information into metavariables for easier subsequent
% changes
tmpX = {handles.CAP1X,handles.CAP2X,handles.CAP3X,handles.CAP4X,handles.CAP5X};
tmpY = {handles.CAP1Y,handles.CAP2Y,handles.CAP3Y,handles.CAP4Y,handles.CAP5Y};
tmpZ = {handles.CAP1Z,handles.CAP2Z,handles.CAP3Z,handles.CAP4Z,handles.CAP5Z};
tmpF = {handles.CAP1_Frames,handles.CAP2_Frames,handles.CAP3_Frames,handles.CAP4_Frames,handles.CAP5_Frames};

%handles.Kmax = handles.Parameters.KMeansClustering.NumberClusters;
% For each CAP...
for i_CAP = 1:handles.Kmax
    
    % Clears the display for each dimension
    cla(tmpX{i_CAP});
    cla(tmpY{i_CAP});
    cla(tmpZ{i_CAP});
    
    % Plots the new slice for each dimension
    tmpX{i_CAP} = plot_slice(handles.CAP(i_CAP,:),...
        get(handles.TVIS_Slider,'Value'),1.5,handles.mask{1},handles.brain,handles.brain_info{1},...
        'X',get(handles.CAP_SliderX,'Value'),tmpX{i_CAP});

    tmpY{i_CAP} = plot_slice(handles.CAP(i_CAP,:),...
        get(handles.TVIS_Slider,'Value'),1.5,handles.mask{1},handles.brain,handles.brain_info{1},...
        'Y',get(handles.CAP_SliderY,'Value'),tmpY{i_CAP});
    
    tmpZ{i_CAP} = plot_slice(handles.CAP(i_CAP,:),...
        get(handles.TVIS_Slider,'Value'),1.5,handles.mask{1},handles.brain,handles.brain_info{1},...
        'Z',get(handles.CAP_SliderZ,'Value'),tmpZ{i_CAP});

    % Sets the frame percentage text visible and at the right value (number
    % of frames from a CAP/total frame number, and then percentage that it
    % stands for)
    set(tmpF{i_CAP},'Visible','on');
    set(tmpF{i_CAP},'String',{[num2str(sum(handles.idx(:)==i_CAP)),'/',...
        num2str(size(handles.idx,1))],[sprintf('%.2f',...
        sum(handles.idx(:)==i_CAP)/size(handles.idx(:),1)*100),' [%]']});
end

guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function Zscore_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Zscore (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% SECTION 4: MISCELLANEOUS %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% General utilities
% Resets the display of a graph object
function handles = ResetGraphDisplay(Graph,handles)

cla(Graph);
set(Graph,'Visible','off');


% --- Executes when figure1 is resized.
function figure1_SizeChangedFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

function handles = ClearSection(eventdata, handles)

% Puts back the seed buttons information to original state
%set(handles.FiguresButton,'Enable','off');
set(handles.ShowCAPsButton,'Enable','off');

% Removes graph display for CAPs
cla(handles.CAP1X);
cla(handles.CAP1Y);
cla(handles.CAP1Z);
cla(handles.CAP2X);
cla(handles.CAP2Y);
cla(handles.CAP2Z);
cla(handles.CAP3X);
cla(handles.CAP3Y);
cla(handles.CAP3Z);
cla(handles.CAP4X);
cla(handles.CAP4Y);
cla(handles.CAP4Z);
cla(handles.CAP5X);
cla(handles.CAP5Y);
cla(handles.CAP5Z);
% Remove graph display pie charts
cla(handles.pie1);
cla(handles.pie2);
cla(handles.pie3);
cla(handles.pie4);
cla(handles.pie5);
set(handles.CAP1_Frames,'Visible','off');
set(handles.CAP2_Frames,'Visible','off');
set(handles.CAP3_Frames,'Visible','off');
set(handles.CAP4_Frames,'Visible','off');
set(handles.CAP5_Frames,'Visible','off');
%Remove graph display Seed names
set(handles.PIE_S1,'Visible','off');
set(handles.PIE_S2,'Visible','off');
set(handles.PIE_S3,'Visible','off');
set(handles.PIE_S1S2,'Visible','off');
set(handles.PIE_S1S3,'Visible','off');
set(handles.PIE_S2S3,'Visible','off');
set(handles.PIE_S1S2S3,'Visible','off');

%Remove CAP Slider
set(handles.CAP_SliderX,'Visible','off');
cla(handles.CAP_SliderX);
set(handles.CAP_SliderY,'Visible','off');
cla(handles.CAP_SliderY);
set(handles.CAP_SliderZ,'Visible','off');
cla(handles.CAP_SliderZ);
set(handles.CAP_XC,'Visible','off');
set(handles.CAP_YC,'Visible','off');
set(handles.CAP_ZC,'Visible','off');

%Remove T slider
set(handles.TVIS,'Visible','off');
set(handles.TVIS_Slider,'Visible','off');
cla(handles.TVIS_Slider);
cla(handles.ColorbarCAP);
set(handles.ColorbarCAP,'Visible','off');

%Remove Similarity Matrix
cla(handles.CAP_Mat);
cla(handles.ColorbarSimMat);
set(handles.ColorbarSimMat,'Visible','off');
set(handles.text_similarity,'Visible','off');
set(handles.Zscore,'Visible','off');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%















