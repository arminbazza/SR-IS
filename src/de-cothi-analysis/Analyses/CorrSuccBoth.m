%% Save dir
save_dir = '/Users/abizzle/Research/SR-IS/figures/';

%% Human Paths
load('Humans/humans.mat')
load('Humans/Q.mat')
load('Humans/MB.mat')
load('Humans/SR.mat')
load('Humans/SR_IS.mat')

%% Human Config Correlations
n_ppt = 18;
ppt_ids = zeros(n_ppt,n_ppt*100);

for i = 1:n_ppt
    tmp_ppt_id = zeros(1,n_ppt);
    tmp_ppt_id(i) = 1;
    ppt_ids(i,:) = reshape(repmat(tmp_ppt_id,100,1),1,[]);
end

ppt_rhos = zeros(n_ppt,4);
for i = 1:n_ppt
    ppt_rhos(i,1) = corr(squeeze(nanmean(Q_succ(ppt_ids(i,:)==1,:,:),[1,3]))', squeeze(nanmean(human_succ(i,:,:),[1,3]))','Type','Spearman');
    ppt_rhos(i,2) = corr(squeeze(nanmean(MB_succ(ppt_ids(i,:)==1,:,:),[1,3]))', squeeze(nanmean(human_succ(i,:,:),[1,3]))','Type','Spearman');
    ppt_rhos(i,3) = corr(squeeze(nanmean(SR_succ(ppt_ids(i,:)==1,:,:),[1,3]))', squeeze(nanmean(human_succ(i,:,:),[1,3]))','Type','Spearman');
    ppt_rhos(i,4) = corr(squeeze(nanmean(SR_imp_succ(ppt_ids(i,:)==1,:,:),[1,3]))', squeeze(nanmean(human_succ(i,:,:),[1,3]))','Type','Spearman');
end

rhos = nanmean(ppt_rhos,1);
rhos_err = nanstd(ppt_rhos,[],1)/sqrt(n_ppt);

figure
hold on
bar(rhos,'LineWidth',2,'EdgeColor','k','FaceColor','w')
errorbar(rhos, rhos_err, '.k', 'LineWidth', 2)

set(gca,'LineWidth',2)
set(gcf,'color','w');
set(gca,'FontSize',18)
set(gca, 'FontName', 'Times New Roman')

title('Humans', 'FontSize', 20, 'FontWeight','normal');
ylabel('Correlation coefficient', 'FontSize', 18);
xticks([1,2,3, 4])
xticklabels({'MF','MB','SR', 'SR-IS'})

% Save human correlation
set(gcf, 'Units', 'inches');
set(gcf, 'Position', [0 0 5 4]);
% exportgraphics(gcf, [save_dir,'corr_human.png'], 'Resolution', 300);

%% Rat Paths
load('Rats/rat.mat')
load('Rats/Q.mat')
load('Rats/MB.mat')
load('Rats/SR.mat')
load('Rats/SR_IS.mat')

rat_SR = SR;
rat_SR_imp = SR_imp;

%% Rat Config Correlation
n_ppt = 9;
ppt_ids = zeros(n_ppt,n_ppt*100);

for i = 1:n_ppt
    tmp_ppt_id = zeros(1,n_ppt);
    tmp_ppt_id(i) = 1;
    ppt_ids(i,:) = reshape(repmat(tmp_ppt_id,100,1),1,[]);
end

ppt_rhos = zeros(n_ppt,4);
for i = 1:n_ppt
    ppt_rhos(i,1) = corr(squeeze(nanmean(Q_succ(ppt_ids(i,:)==1,:,:),[1,3]))', squeeze(nanmean(rat_succ(i,:,:),[1,3]))','Type','Spearman');
    ppt_rhos(i,2) = corr(squeeze(nanmean(MB_succ(ppt_ids(i,:)==1,:,:),[1,3]))', squeeze(nanmean(rat_succ(i,:,:),[1,3]))','Type','Spearman');
    ppt_rhos(i,3) = corr(squeeze(nanmean(SR_succ(ppt_ids(i,:)==1,:,:),[1,3]))', squeeze(nanmean(rat_succ(i,:,:),[1,3]))','Type','Spearman');
    ppt_rhos(i,4) = corr(squeeze(nanmean(SR_imp_succ(ppt_ids(i,:)==1,:,:),[1,3]))', squeeze(nanmean(rat_succ(i,:,:),[1,3]))','Type','Spearman');
end

rhos = nanmean(ppt_rhos,1);
rhos_err = nanstd(ppt_rhos,[],1)/sqrt(n_ppt);

figure
hold on
bar(rhos,'LineWidth',2,'EdgeColor','k','FaceColor','w')
errorbar(rhos, rhos_err, '.k', 'LineWidth', 2)

set(gca,'LineWidth',2)
set(gcf,'color','w');
set(gca,'FontSize',18)
set(gca, 'FontName', 'Times New Roman')

title('Rats', 'FontSize', 20, 'FontWeight','normal', 'FontName', 'Times New Roman');
ylabel('Correlation coefficient', 'FontSize', 18);
xticks([1,2,3, 4])
xticklabels({'MF','MB','SR', 'SR-IS'})

% Save rat correlation
set(gcf, 'Units', 'inches');
set(gcf, 'Position', [0 0 5 4]);
% exportgraphics(gcf, [save_dir,'corr_rat.png'], 'Resolution', 300);