%% Translate AoA and ToF into x and y coordinates.
function [x,y]=com_xy(AOA,TOF,xr)
    dt = TOF * 299792458;  % light speed
    theta = AOA/180*pi; 
    
    % x1 = (dt^2 + xr^2 - (dt*(dt^3*cos(zeta)^2 + dt^3*sin(zeta)^2 ...
    %     + xr^3*sin(zeta)*(cos(zeta)^2 + sin(zeta)^2)^(1/2) - dt*xr^2*cos(zeta)^2 ...
    %     - dt*xr^2*sin(zeta)^2 - dt^2*xr*sin(zeta)*(cos(zeta)^2 ...
    %     + sin(zeta)^2)^(1/2)))/(dt^2*cos(zeta)^2 + dt^2*sin(zeta)^2 - xr^2*sin(zeta)^2))/(2*xr);
    
    x = (dt^2 + xr^2 - (dt*(dt^3*cos(theta)^2 + dt^3*sin(theta)^2 ...
        - xr^3*sin(theta)*(cos(theta)^2 + sin(theta)^2)^(1/2) ...
        - dt*xr^2*cos(theta)^2 - dt*xr^2*sin(theta)^2 ...
        + dt^2*xr*sin(theta)*(cos(theta)^2 + sin(theta)^2)^(1/2)))/(dt^2*cos(theta)^2 ...
        + dt^2*sin(theta)^2 - xr^2*sin(theta)^2))/(2*xr);
    
    % y1 = (xr^2*cos(zeta) - dt^2*cos(zeta) + (dt*cos(zeta)*(dt^3*cos(zeta)^2 ...
    %     + dt^3*sin(zeta)^2 + xr^3*sin(zeta)*(cos(zeta)^2 + sin(zeta)^2)^(1/2) ...
    %     - dt*xr^2*cos(zeta)^2 - dt*xr^2*sin(zeta)^2 - dt^2*xr*sin(zeta)*(cos(zeta)^2 ...
    %     + sin(zeta)^2)^(1/2)))/(dt^2*cos(zeta)^2 + dt^2*sin(zeta)^2 - xr^2*sin(zeta)^2))/(2*xr*sin(zeta));
    
    y = (xr^2*cos(theta) - dt^2*cos(theta) + (dt*cos(theta)*(dt^3*cos(theta)^2 ...
        + dt^3*sin(theta)^2 - xr^3*sin(theta)*(cos(theta)^2 + sin(theta)^2)^(1/2) ...
        - dt*xr^2*cos(theta)^2 - dt*xr^2*sin(theta)^2 + dt^2*xr*sin(theta)*(cos(theta)^2 ...
        + sin(theta)^2)^(1/2)))/(dt^2*cos(theta)^2 + dt^2*sin(theta)^2 - xr^2*sin(theta)^2))/(2*xr*sin(theta));
end
