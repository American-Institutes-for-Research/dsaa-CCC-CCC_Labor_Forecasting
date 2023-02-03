% Generate density forecast figures in Appendix A

v_datenum2 = [v_datenum; v_datenum(end)+(1:H)'];
grayshade = .5*ones(1,3);
graph_size = [4 3];
SFont = 12;

filename_country = v_country; 
for i = 1:N
    filename_country{i} = erase(v_country{i},' ');
end

%% Loop over countries/regions
for i = 1:N      
    F1 = figure;
    hold on

    % Plot in-sample one-step-ahead predictions and point forecasts 
    qfcst = quantile([squeeze(in_smpl_draw(startv(i):end,i,ix_out));...
            squeeze(out_smpl_draw(:,i,ix_out))]',[.1 .2 .5 .8 .9]);
    plot(v_datenum2(startv(i):end),qfcst(3,:),'k-','linewidth',1)
    
    % Plot actual daily infections
    scatter(v_datenum(startv(i):end),Y_level(startv(i)+2:end,i),20,'k','LineWidth',1)
    
    % Plot density forecasts (quantiles of posterior predictive distribution)
    jbfill(v_datenum2(T+1:end)',qfcst(5,end-H+1:end),qfcst(1,end-H+1:end),grayshade,grayshade,1,0.25);
    jbfill(v_datenum2(T+1:end)',qfcst(4,end-H+1:end),qfcst(2,end-H+1:end),grayshade,grayshade,1,0.25);
    
    % Plot a verticle line indicating forecast origin
    xline(v_datenum(end),'k--');
    hold off
    
    datetick('x','mm/dd')
    xtickangle(45)
    xlim(v_datenum2([startv(i) end]))
    title(v_country{i},'FontSize',24,'FontWeight','bold')
    
    % Setup graph formats and save the graph as .png and .fig files
    graph_tag = [tags{2},filename_country{i},tags{3}];
    graph_out(F1,graph_tag,[],SFont,graph_size,[]);
    if mod(i,5) == 0
        close all
    end
end
close all