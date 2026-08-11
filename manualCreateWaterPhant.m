PH.xbnds = [-19.95:0.3:19.95];
PH.ybnds = [0,0.25:0.3:11, 35];
PH.zbnds = [-17.55:0.3:17.55];

PH.med = ones(length(PH.xbnds),length(PH.ybnds),length(PH.zbnds))
PH.rhor = ones(length(PH.xbnds),length(PH.ybnds),length(PH.zbnds))

PH.media = {'H2O700ICRU'}



Plan_Dir = pwd;
PhantName = fullfile(Plan_Dir,['TG51.egsphant']);
write_egsphant_CN(PhantName,PH);