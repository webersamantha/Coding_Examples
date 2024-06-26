%% This function plots the CAPs obtained for a given case
% Inputs:
% - Cp: a matrix of size n_clusters x n_voxels with the CAPs (or patterns)
% to plot - Cp = CAP(1,:);
% - T: the threshold below which voxels will not be colored - T = 0.5;
% - maxC: the maximal value at which the color display will saturate -
% maxC= 1
% - mask: a very long vector of logicals symbolizing the regions of the
% mask = mask{1}
% brain that are actually to be considered (in-brain voxels typically)
% - brain_final: a 3D matrix used to plot the greyscale brain template on which
% to overlay activity patterns - brain_final = brain;
% - ai: the nii data related to the considered seed, including information
% on the scale differences between Cp indexes ('distance between voxels of
% the matrix') and actual MNI space - ai = brain_info{1};
% - Dimension: the type of slice to plot ('X', 'Y' or 'Z') Dimension = 'X'
% - MNI: the MNI coordinate of the slice to plot [X Y Z] - only one! e.g.,
% MNI = [5]
% - Handle: the handle of the graph to update
function [Handle] = plot_slice_SW(Cp,T,maxC,mask,brain_final,ai,Dimension,MNI,Handle)

    % Computes the matrix index matching the MNI coordinates of interest.
    Map = inv(ai.mat);
    
    switch Dimension
        case 'X'
            ctr = round(Map(1,1)*MNI+Map(1,4));     
        case 'Y'
            ctr = round(Map(2,2)*MNI+Map(2,4));
        case 'Z'
            ctr = round(Map(3,3)*MNI+Map(3,4));
    end
    
    % temp contains the volume values (within mask), and is a 3D volume after
    % those lines
    temp = nan(size(mask)); 
    temp(mask) = Cp(1,:);
    temp(isnan(temp)) = 0;
    temp = reshape(temp,ai.dim);
    
    % ho contains the structural underlay slice to plot at right slice,
    % while tmpp contains the values to plot on top
    switch Dimension
        case {'X'}
            tmpp = squeeze(temp(ctr,:,:));
            ho = squeeze(brain_final(ctr,:,:)); 
        case {'Y'}
            tmpp = squeeze(temp(:,ctr,:));
            ho = squeeze(brain_final(:,ctr,:)); 
        case {'Z'}
            tmpp = squeeze(temp(:,:,ctr));
            ho = squeeze(brain_final(:,:,ctr)); 
    end

    % I is an image with values from 0 to 1 (because the original
    % image had no negative value)
    %I = double(ho)/max(double(ho(:)));
    I = double(ho)'/max(double(ho(:)));
    I(I==0) = 1;

    % Creates an 'image with three values per pixel'
    Irgb = cat(3,I,I,I);

    % Actual plotting
    imagesc(Irgb); 
    hold(Handle,'on');
    hold('on')
    % Plots the slice of interest on top of the brain template
    h=imagesc(tmpp','Parent',Handle); 
    h = imagesc(tmpp');
    set(Handle,'YDir','normal'); 

    % At this stage, ddd contains a set of colors, with white if the values
    % are too low. We ask the axes of interest to be using this colormap
    % colormap(Handle,ddd);
    tmp_cm = cbrewer('div','RdBu',1000);
    tmp_cm(tmp_cm < 0) = 0;
    colormap(Handle,flipud(tmp_cm));
    colormap(flipud(tmp_cm));
    % Defines that the topmost and bottommost elements of the
    % colormap will map maxC and -maxC respectively
    caxis(Handle,[-1 1]*maxC); 
    caxis([-1 1]*maxC);
    % Opacity: sets data points below the value of interest as
    % transparent (they are white, but transparent); note that we do this
    % specifically for the h imagesc (the one to plot on top)
    A = ones(size(tmpp)); 
    A(abs(tmpp)<T)=0;
    set(h,'AlphaData',A');
    
    set('YDir','normal'); 
    axis('square'); 
    axis('off');
end