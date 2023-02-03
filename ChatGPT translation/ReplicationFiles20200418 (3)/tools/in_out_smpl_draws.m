% Produce in-sample one-step-ahead predictions and out-of-sample density forecasts 
% based on MCMC draws of parameters

ix_out = B_draw+(1:N_draw);

startv = nan(1,N); % start of the unbalanced sample for each country/region, calendar time
t_trimmed = nan(2,N); % trimmed observations (due to limited-info analysis), calendar time
for i = 1:N  
    startv(i) = find(ix1(:,i)==1,1,'first');
    aux = isnan(ix1(startv(i):end,i));
    if sum(aux)>0
        t_trimmed(:,i) = [find(aux,1,'first') find(aux,1,'last')]+startv(i)-1;
    end
end
t_star_draw = startv-1-Lamb_draw(:,:,1)./Lamb_draw(:,:,2); % t*, calendar time

Coef_aux_draw = nan(size(Lamb_draw)); % transform coefficients for easier implementation
Coef_aux_draw(:,:,1) = Lamb_draw(:,:,1).*(1-Coef_homog_draw)+Lamb_draw(:,:,2).*Coef_homog_draw;
Coef_aux_draw(:,:,2) = Lamb_draw(:,:,2).*(1-Coef_homog_draw);
Coef_aux_draw(:,:,3) = Lamb_draw(:,:,3).*(1-Coef_homog_draw)+Lamb_draw(:,:,4).*Coef_homog_draw;
Coef_aux_draw(:,:,4) = Lamb_draw(:,:,4).*(1-Coef_homog_draw);

%% In-sample one-step-ahead predictions
in_smpl_draw = nan(T,N,B_draw+N_draw);
for i_draw = ix_out
    for i = 1:N
        if ~isnan(t_trimmed(1,i)) % if peaked before T, contruct xt*1(t>t*) accordingly
            aux_trim = t_trimmed(1,i):t_trimmed(2,i);
            if t_star_draw(i_draw,i)<t_trimmed(1,i) 
                XX_heter(aux_trim,i,3:4) = XX_heter(aux_trim,i,1:2);
            elseif t_star_draw(i_draw,i)>=t_trimmed(2,i)
                XX_heter(aux_trim,i,3:4) = 0;
            else
                t_aux = floor(t_star_draw(i_draw,i));
                XX_heter(t_trimmed(1,i):t_aux,i,3) = 0;
                XX_heter(t_aux+1:t_trimmed(2,i),i,3) = 1;
                XX_heter(aux_trim,i,4) = XX_heter(aux_trim,i,2).*XX_heter(aux_trim,i,3);
            end
        end
    end
    aux_heter = zeros(T,N);
    for i_heter = 1:n_heter
        aux_heter = aux_heter+Coef_aux_draw(i_draw,:,i_heter).*XX_heter(:,:,i_heter);
    end
    in_smpl_draw(:,:,i_draw) = exp(Coef_homog_draw(i_draw)*XX_homog+aux_heter...
        +sqrt(sigma2_draw(i_draw,:)).*randn(T,N)).*Y_level(2:end-1,:);
end

%% Out-of-sample forecasts 
XX_heter2 = nan(H,N,n_heter);
XX_heter2(:,:,1) = 1;
for i = 1:N
    XX_heter2(:,i,2) = XX_heter(end,i,2)+(1:H);
    if t_trimmed(1,i)+5<=T % if peaked before T, contruct xt*1(t>t*) accordingly
        XX_heter2(:,i,3:4) = XX_heter2(:,i,1:2);
    end
end

out_smpl_gr_draw = nan(H,N,B_draw+N_draw);
for i_draw = ix_out
    for i = 1:N
        if isnan(t_trimmed(1,i)) || t_trimmed(1,i)+5>T 
            % if peaked on/after T, contruct xt*1(t>t*) accordingly
            if t_star_draw(i_draw,i)<T+1 
                XX_heter2(:,i,3:4) = XX_heter2(:,i,1:2);
            elseif t_star_draw(i_draw,i)>=T+H
                XX_heter2(:,i,3:4) = 0;
            else
                t_aux = floor(t_star_draw(i_draw,i));
                XX_heter2(1:t_aux-T,i,3) = 0;
                XX_heter2(t_aux-T+1:end,i,3) = 1;
                XX_heter2(:,i,4) = XX_heter2(:,i,2).*XX_heter2(:,i,3);
            end
        end
    end
    out_smpl_gr_draw(1,:,i_draw) = Coef_homog_draw(i_draw)*YY(end,:)...
        +sqrt(sigma2_draw(i_draw,:)).*randn(1,N);
    for i_heter = 1:n_heter
        out_smpl_gr_draw(1,:,i_draw) = out_smpl_gr_draw(1,:,i_draw)...
            +Coef_aux_draw(i_draw,:,i_heter).*XX_heter2(1,:,i_heter);
    end
    for h = 2:H
        out_smpl_gr_draw(h,:,i_draw) = Coef_homog_draw(i_draw)*out_smpl_gr_draw(h-1,:,i_draw)...
            +sqrt(sigma2_draw(i_draw,:)).*randn(1,N);
        for i_heter = 1:n_heter
            out_smpl_gr_draw(h,:,i_draw) = out_smpl_gr_draw(h,:,i_draw)...
                +Coef_aux_draw(i_draw,:,i_heter).*XX_heter2(h,:,i_heter);
        end
    end  
end
 
out_smpl_draw = exp(cumsum(out_smpl_gr_draw)).*Y_level(end,:);

%% Housekeeping
clear Coef_aux_draw;