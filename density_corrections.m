clc;
clear all;
%%
DNS = csvread('/home/simon/Desktop/Bachelor_project/data/DNS_matlab.csv',1,0);
%%
altitude = DNS(:,1);
latitude = DNS(:,2);
longitude = DNS(:,3);
year = 2015;
dayOfYear= DNS(:,12);
UTseconds = DNS(:,13);
f107Average = DNS(:,11);
f107Daily = DNS(:,10);
magneticIndex = 4;

% get the model for the sat
[~,rho_sat] = atmosnrlmsise00(altitude, latitude, longitude, year, dayOfYear, UTseconds, f107Average, f107Daily,4);

%% Altitude
std_altitude = 470*10^3;
[~,rho_std] = atmosnrlmsise00(std_altitude, latitude, longitude, year, dayOfYear, UTseconds, f107Average, f107Daily,4);

cFactor_alt = rho_std(:,6)./rho_sat(:,6);

%% F10.7
std_f107 = 135;
[~,rho_std] = atmosnrlmsise00(altitude, latitude, longitude, year, dayOfYear, UTseconds, std_f107, std_f107,4);

cFactor_f10 = rho_std(:,6)./rho_sat(:,6);

%%
figure(1)
scatter(altitude(1:100:end),cFactor_alt(1:100:end))
title('Correction Factor Altitude');

figure(2)
scatter((f107Average(1:100:end)+f107Daily(1:100:end))/2,cFactor_f10(1:100:end))
title('Correction Factor F10');

%%
Cfactors=[cFactor_alt,cFactor_f10];
csvwrite('/home/simon/Desktop/Bachelor_project/data/Correction_factors.csv',Cfactors)
