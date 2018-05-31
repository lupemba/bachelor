clc
%% nrlmsise test
altitude = [300 300 500]*10^3;
latitude = [40 40 35];
longitude = [-34 360-34 0];
year = 2015;
dayOfYear= [1,1,3];
UTseconds = [100,100,100];
f107Average = [80,80,80];
f107Daily = [100,100,70];
magneticIndex = 4;

[T rho] = atmosnrlmsise00(altitude, latitude, longitude, year, dayOfYear, UTseconds, f107Average, f107Daily,4)

%% 
atmosnrlmsise00( 300000, 45, -50, 2007, 4, 0, 'Oxygen')
[T rho] = atmosnrlmsise00( 300000, 45, -50, 2007, 4, 0, 'Oxygen')