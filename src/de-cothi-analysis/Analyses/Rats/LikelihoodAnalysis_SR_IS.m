load('likelihoods_individuals.mat')
load('likelihoods_RW.mat')
load('rat_SR_IS_llik.mat')

% Set default properties for all figures
set(groot, 'defaultAxesFontName', 'HelveticaNeue')
set(groot, 'defaultTextFontName', 'HelveticaNeue')
set(groot, 'defaultAxesFontWeight', 'normal')
set(groot, 'defaultTextFontWeight', 'normal')
set(groot, 'defaultAxesXColor', 'k')
set(groot, 'defaultAxesYColor', 'k')
set(groot, 'defaultAxesZColor', 'k')
set(groot, 'defaultTextColor', 'k')

cmap = brewermap(6,'Set1');
lliks = zeros([4,size(MB_acts)]);
lliks(1,:,:,:) = cellfun(@sum,Q_llik);
lliks(2,:,:,:) = cellfun(@sum,MB_llik);
lliks(3,:,:,:) = cellfun(@sum,SR_llik);
lliks(4,:,:,:) = cellfun(@sum,SR_IS_llik); % New SR-IS model
% Sum across starting points (dim 4) and mazes (dim 3) for each subject
lliks_per_subject = squeeze(sum(sum(lliks, 4), 3));  % 4 x 9
ll_winners = lliks == max(lliks,[],1);
baseline = sum(cellfun(@sum,RW_llik),'all');
ll = sum(lliks,[2,3,4]);
hybrid = max(cellfun(@nansum,MB_llik),cellfun(@nansum,Q_llik));
lliks = zeros([4,size(MB_acts)]);
lliks_RW = exp(cellfun(@mean,RW_llik));
lliks(1,:,:,:) = exp(cellfun(@mean,Q_llik)) - lliks_RW;
lliks(2,:,:,:) = exp(cellfun(@mean,MB_llik)) - lliks_RW;
lliks(3,:,:,:) = exp(cellfun(@mean,SR_llik)) - lliks_RW;
lliks(4,:,:,:) = exp(cellfun(@mean,SR_IS_llik)) - lliks_RW;


%% Model Selection
% Number of parameters per model
k_MB = 1;
k_SR = 2;
k_MF = 2;
k_SR_IS = 3;
n_models = 4;
n_subjects = 9;

% Calculate BIC for each model for each subject
model_names = {'MF', 'MB', 'SR', 'SR-IS'};
n_params = [k_MF, k_MB, k_SR, k_SR_IS]';
n_datapoints = 25 * 10;
BIC = -2*lliks_per_subject + n_params .* log(n_datapoints);
lme = lliks_per_subject - 0.5*(n_params .* log(n_datapoints));

% Use log model evidence in CBM
[alpha,mf,xp,pxp,bor,g] = cbm_spm_BMS(lme');

% Compare models
[~, best_model_per_subject] = min(BIC, [], 1);
group_BIC = sum(BIC, 2);
[~, best_group_model] = min(group_BIC);

% Calculate delta BIC relative to best model per subject
delta_BIC = BIC - min(BIC, [], 1);  % 4 x 9
mean_delta_BIC = mean(delta_BIC, 2);

% Print results
fprintf('\n=== MODEL COMPARISON RESULTS ===\n');
fprintf('Best group-level model: %s\n\n', model_names{best_group_model});
model_counts = histcounts(best_model_per_subject, 0.5:(n_models+0.5));
fprintf('Number of subjects best fit by each model:\n');
for i = 1:n_models
    fprintf('  %s: %d subjects (%.1f%%)\n', ...
            model_names{i}, model_counts(i), pxp(i));
end

fprintf('\n')
fprintf('CBM fitting results:\n');
for i = 1:n_models
    fprintf('  %s - model frequency: %.2f%% | protected xp: %.2f%%)\n', ...
            model_names{i}, mf(i), ...
            100*model_counts(i)/n_subjects);
end

% fprintf('\nMean Δ BIC (relative to best model):\n');
% model_names = {'MF', 'MB', 'SR', 'SR-IS'};
% for i = 1:4
%     fprintf('  %s: %.2f\n', model_names{i}, mean_delta_BIC(i));
% end
% 
% % Also check group-level differences
% fprintf('\nGroup BIC differences from SR_IS:\n');
% for i = 1:4
%     fprintf('  %s: %.2f\n', model_names{i}, group_BIC(i) - group_BIC(4));
% end


%% Figure 1: Overall Model Comparison
figure
hold on
set(gca,'FontSize',18, 'FontName', 'HelveticaNeue')
bar(1:4,ll,'FaceColor','k')
title('Rats', 'FontName', 'Times New Roman','Color','Black')
xticks(1:4)
xticklabels({'MF','MB','SR','SR-IS'})
box on
ylabel('Log Likelihood', 'FontName', 'HelveticaNeue')
yline(baseline, 'Color','red', 'LineWidth',2,'LineStyle','--')
ylim([-5.6e4,-5.25e4])
yticks([-5.5e4, -5.25e4])
set(gcf,'color','w');
set(gca,'LineWidth',2)


%% Figure 2: Early vs Late Trials
% Split data into early trials (1-5) and late trials (6-10)
trial_lliks1 = squeeze(nanmean(lliks(:,:,:,1:5),[3,4]));  % Early trials
trial_lliks2 = squeeze(nanmean(lliks(:,:,:,6:10),[3,4])); % Late trials

figure
hold on
set(gca, 'Color', 'w')
bar([1:4, 5.5:1:8.5], [mean(trial_lliks1, 2); mean(trial_lliks2,2)],'EdgeColor','k','FaceColor','w','LineWidth',2)

for i = 1:9
    plot((1:4) + (-0.05+0.1*rand(1,4)), trial_lliks1(:,i), 'k.-','MarkerSize',12, 'linewidth',1.5)
    plot((5.5:1:8.5) + (-0.05+0.1*rand(1,4)), trial_lliks2(:,i), 'k.-','MarkerSize',12, 'linewidth',1.5)
end

title('Rats')
xline(4.75,':k','LineWidth',2)
set(gca,'FontSize',18)
set(gcf,'color','w');
set(gca,'LineWidth',2)
ylim([0,0.042])
yticks([0, 0.03])
xticks([1:4, 5.5:1:8.5])
xticklabels({'MF', 'MB','SR','SR-IS','MF', 'MB','SR','SR-IS'})
ylabel('Avg action likelihood', 'FontName', 'HelveticaNeue')
box on


%% Figure 3: Heatmap by Maze Configuration
figure('Position', [100, 100, 800, 250])
imagesc(squeeze(nanmedian(lliks,[2,4])))
% set(gca, 'FontWeight', 'normal')
% set(gca, 'FontName', 'HelveticaNeue')
set(gca,'FontSize',20)
set(gcf,'color','w');
set(gca,'LineWidth',2)
yticks([1,2,3,4])
yticklabels({'MF','MB','SR','SR-IS'})
colormap jet
cb = colorbar;
% set(cb, 'FontWeight', 'normal')
% set(cb, 'FontName', 'HelveticaNeue')
caxis([0.0,0.03])
pbaspect([25 4 1])

set(gca, 'FontWeight', 'normal')
set(gca, 'TickDir', 'out')
set(gca, 'TickLength', [0 0])
set(cb, 'Ticks', [0, 0.03])
set(cb, 'TickLabels', {'0', '0.03'})
set(cb, 'Color', 'k')
set(cb, 'FontSize', 20)


%% Figure 4: Winner Proportion Pie Chart with percentages
figure
pie_data = nanmean(ll_winners,[2,3,4]) / sum(nanmean(ll_winners,[2,3,4]));
labels = {'MF', 'MB', 'SR', 'SR-IS'};
pie_labels = cell(1,4);
for i = 1:4
    pie_labels{i} = sprintf('%s\n%.1f%%', labels{i}, pie_data(i)*100);
end
h = pie(pie_data, pie_labels);
% Make the pie chart lines thicker and black
for i = 1:2:length(h)
    set(h(i), 'LineWidth', 3, 'EdgeColor', 'k')
end
colormap(ones(3))
set(gca,'FontSize',18)
set(gca,'color','b')
set(gcf,'color','w');
set(gca,'LineWidth',4)