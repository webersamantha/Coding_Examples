%% Script to visualize Consensus Matrices
% Samantha Weber June 2022

% Adapt k accordingly
function ConsensusMatrixPlot_SW(Consensus, SavePath, fancyName, K_range)      
        for k = 1:(size(K_range,2))
            imagesc(Consensus(:,:,k));
            xlabel('Frames');
            ylabel('Frames');
            title(['K = ' num2str(k + 1)]);
            axis('square','on');
            saveas(gcf,fullfile(SavePath,fancyName,['ConsensusMatrix_K_' num2str(k + 1) '.fig']));
            saveas(gcf,fullfile(SavePath,fancyName,['ConsensusMatrix_K_' num2str(k + 1) '.jpg']));
        end
   close;     
end