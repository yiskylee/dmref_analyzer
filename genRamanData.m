function summary = genRamanData(inputPath)
    %%%%%%%%%%%%%% Read teh csv file
    data = csvread(inputPath);

    %%%%%%%%%%%%%% D band and G band area
    PartionPoint1 = find(data(:,1)>600);   %
    PartionPoint1 = PartionPoint1(1);
    PartionPoint2 = find(data(:,1)<2200);   %
    PartionPoint2 = PartionPoint2(end);
    PartionPoint3 = find(data(:,1)<1000);   %
    PartionPoint3 = PartionPoint3(end);
    PartionPoint4 = find(data(:,1)>1800);   %
    PartionPoint4 = PartionPoint4(1);
    DGData=data(PartionPoint1:PartionPoint2,:);
    w=[];
    I=[];
    Icorr=[];
    subI1 =[];
    subw1 =[];
    ww=[];
    II=[];
    IIcorr=[];
    subII1 =[];
    subww1 =[];
    w = DGData(:,1);
    I = DGData(:,2);
    Icorr=I;
    subI1 = [I(1:PartionPoint3-PartionPoint1); I(PartionPoint4-PartionPoint1:end)]; subw1 = [w(1:PartionPoint3-PartionPoint1); w(PartionPoint4-PartionPoint1:end)];
    [P1,S1] = polyfit(subw1,subI1,3); % returns the polynomial coefficients p and a structure S for use with polyval to obtain error estimates or predictions.
    Icorr(1:length(w)) = I - polyval(P1,w(:)); % Polynomial evaluation, so we can produce a baseline according to the polynomial from the sections where no peaks exist
    AAA=Icorr;
    %%%%%%%%%%%%%% G prone data or 2D data
    PartionPoint5 = find(data(:,1)>2000);   %
    PartionPoint5 = PartionPoint5(1);
    PartionPoint6 = find(data(:,1)<3000);   %
    PartionPoint6 = PartionPoint6(end);
    PartionPoint7 = find(data(:,1)<2500);   %
    PartionPoint7 = PartionPoint7(end);
    PartionPoint8 = find(data(:,1)>2800);   %
    PartionPoint8 = PartionPoint8(1);
    G2Data=data(PartionPoint5:PartionPoint6,:);
    ww = G2Data(:,1);
    II = G2Data(:,2);
    IIcorr=II;
    subII1 = [II(1:PartionPoint7-PartionPoint5); II(PartionPoint8-PartionPoint5:end)]; subww1 = [ww(1:PartionPoint7-PartionPoint5); ww(PartionPoint8-PartionPoint5:end)];
    [PP1,SS1] = polyfit(subww1,subII1,3);% returns the polynomial coefficients p and a structure S for use with polyval to obtain error estimates or predictions.
    IIcorr(1:length(ww)) = II(1:length(ww)) - polyval(PP1,ww(:)); %Polynomial evaluation, so we can produce a baseline according to the polynomial from the sections where no peaks exist
    BBB=IIcorr;
    %%%%%%%%%%%%%%% lorentzian curve fitting
    % Find the peak information
    % PEAKFIT(signal,wavenumber center,window,Numb of Peaks,peakshape,extra,trial numb,start guess);
    [DFitResults,DLowestError,DBestStart,Dxi,Dyi,Dyoffset] = peakfit([w(:,1) AAA(:,1)],1350.1,100,2,2,0,30);
    [GFitResults,GLowestError,GBestStart,Gxi,Gyi,Gyoffset] = peakfit([w(:,1) AAA(:,1)],1576.3,120,2,2,0,30);
    [FitResults,LowestError,BestStart,xi,yi,yoffset] = peakfit([ww(:,1) BBB(:,1)],2685.1,120,2,2,0,30);
    % Specifies the peak shape of the model: "peakshape" = 1-5.
    % (1=Gaussian (default), 2=Lorentzian, 3=logistic, 4=Pearson, and 5=exponentionally broadened Gaussian)
    % Specifies the value of 'extra', used in the Pearson and the exponentionally broadened Gaussian shapes to fine-tune the peak shape.
    % Performs "NumTrials" trial fits and selects the best one (with lowest fitting error). NumTrials can be any positive integer (default is 1).
    %%%%% D Peak Lorentzian fitting
    Dx = 1260:0.01:1420;
    Dy=[];
    Dy1=[];
    Dy2=[];
    for k1=1:length(Dx)
        Dy1(k1)=DFitResults(1,3)/(1+((Dx(k1)-DFitResults(1,2))/(DFitResults(1,4)/2))^2); % Peak 1
        Dy2(k1)=DFitResults(2,3)/(1+((Dx(k1)-DFitResults(2,2))/(DFitResults(2,4)/2))^2); % peak 2
        Dy(k1) = Dy1(k1) + Dy2(k1) + Dyoffset;
    end;
    %The "FitResults" are, from left to right, peak number, peak position, peak intensity, peak width, and peak area.
    % G Peak Lorentzian fitting
    Gx = 1470:0.01:1670;
    Gy=[];
    Gy1=[];
    Gy2=[];
    for k2=1:length(Gx)
        Gy1(k2)=GFitResults(1,3)/(1+((Gx(k2)-GFitResults(1,2))/(GFitResults(1,4)/2))^2);
        Gy2(k2)=GFitResults(2,3)/(1+((Gx(k2)-GFitResults(2,2))/(GFitResults(2,4)/2))^2);
        Gy(k2) = Gy1(k2) + Gy2(k2) + Gyoffset;
    end;
    % 2D Peak Lorentzian fitting
    x = 2560:0.01:2800;
    G2y=[];
    G2y1=[];
    G2y2=[];
    for k3=1:length(x)
        G2y1(k3)=FitResults(1,3)/(1+((x(k3)-FitResults(1,2))/(FitResults(1,4)/2))^2); % Peak 1
        G2y2(k3)=FitResults(2,3)/(1+((x(k3)-FitResults(2,2))/(FitResults(2,4)/2))^2); % peak 2
        G2y(k3) = G2y1(k3) + G2y2(k3) + yoffset;
    end;
    %%%%%%%%%%%%%% Output Table
    [Dmax Dmaxindex] = max(Dy);
    [Gmax Gmaxindex] = max(Gy);
    [G2max, G2maxindex] = max(G2y);
    % summary table
    summary=[Dmax/Gmax,G2max/Gmax,Dx(Dmaxindex),Gx(Gmaxindex),x(G2maxindex)];