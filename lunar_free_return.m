% Federico Spada --- 2023/11/21

clear;

% provide path of installation of the MICE library (MATLAB interface of SPICE)
addpath('/Users/fs255/matlab/mice/lib/','/Users/fs255/matlab/mice/src/mice/');
% provide path of required Kernels via meta-Kernel file "spice.mkn"
cspice_furnsh('spice.mkn');

global et0 mu_e mu_m R_e R_m days xf

% constants
days = 86400;    % days to seconds conversion factor 
mu_e = 398600.4; % Earth gravity parameter (km^3/s^2)
mu_m = 4902.8;   % Moon gravity parameter (km^3/s^2)
R_e = 6378;      % radius of the Earth (Km) 
R_m = 1737;      % radius of the Moon (km) 
R_S = 66183;     % radius of the SOI of the Moon (km)


% for the numerical parameters in the following, cf.: 
% Ocampo, C., 2010, "Elements of a Software System for Spacecraft Trajectory 
% Optimization" in Conway, B.A. ed., 2010. Spacecraft trajectory optimization 
% (Vol. 29). Cambridge University Press.

% fixed parameters
xf = [0.0, R_e+200, 0.0, 0.0, R_e+200, 0.0, 0.0, R_m+8263, 0.0];

% reference initial epoch
epoch0 = '2009-4-3 00:00:00 UTC';
et0 = cspice_str2et(epoch0);

% initial values of the search parameters
x0 = [ 1.0, 0.0, 0.0, 0.0, 3.2, ...
       8.0, -1.0, 45.0, 0.0, 0.0,-3.2, ...
       4.0, 1.0, 1.0, 180.0, 180.0, 0.0, ...
      -1.0 ];

% steps for numerical gradient calculation
delta = [ 1e-4, 1e-3, 1e-3, 1e-3, 1e-6, ...
          1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-6, ...
          1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, ...
          1e-4 ];

%%% solve optimization problem
% optimization options
options = optimoptions('fmincon', 'Display', 'iter', ...
'Algorithm', 'interior-point',...
'FiniteDifferenceStepSize',delta, ...
'FiniteDifferenceType','forward');
%
% set up problem
problem.options = options;
problem.solver = 'fmincon';
problem.objective = @objective_function;
problem.x0 = x0;
problem.nonlcon = @constraints;
% run optimization
x = fmincon(problem);

% screen output
display_x(x,xf)

% propagation with optimized parameters
[t1, y1, t2, y2, t3, y3, t4, y4] = propagate(x);
ns = length(t1) + length(t2) + length(t3) + length(t4);
ts = [t1; t2; t3; t4];
ys = [y1; y2; y3; y4];
is = [1*ones(length(t1),1); 2*ones(length(t2),1); ...
      3*ones(length(t3),1); 4*ones(length(t4),1)];

% 3D plot of the trajectory, inertial frame
figure(1)
clf
% Earth
plot3(0,0,0,'b.','markersize',10)
hold on
% optimized trajectory
for i=1:4
    ii = find(is == i);
    plot3(ys(ii,1),ys(ii,2),ys(ii,3),'.-')
end
% lunar orbit
tm = linspace(0,x(6))*days;
[rm_, ~] = cspice_spkpos('301', et0+tm, 'J2000', 'NONE', '399');
plot3(rm_(1,:),rm_(2,:),rm_(3,:),'-','color',[.7 .7 .7])
% SoI of the Moon
[X,Y,Z] = sphere(20);
[rmo_, ~] = cspice_spkpos('301', et0+x(12)*days, 'J2000', 'NONE', '399');
mesh(R_S*X + rmo_(1),R_S*Y + rmo_(2),R_S*Z + rmo_(3),...
'FaceAlpha','0.2','EdgeColor','k');
%
hold off
grid on
xlabel('X (km)')
ylabel('Y (km)')
zlabel('Z (km)')
axis equal
legend('','S1: outbound','S2: inbound',...
'S3: Moon departure','S4: Moon approach','Orbit of the Moon')

% 3D plot of the trajectory, synodic frame
figure(2)
clf
% Earth
plot3(0,0,0,'b.','markersize',10)
hold on
% optimized trajectory
[svm, lt] = cspice_spkezr('301', et0+ts', 'J2000', 'NONE', '399');
rm_ = svm(1:3,:);
n = length(ys);
ys1 = zeros(n,3);
rm1 = zeros(3,n);
for i=1:n
    rmi_ = svm(1:3,i);
    vmi_ = svm(4:6,i);
    i_ = rmi_/norm(rmi_);
    k_ = cross(rmi_,vmi_)/norm(cross(rmi_,vmi_));
    j_ = cross(k_,i_);
    Q = [i_ j_ k_]';
    r_ = ys(i,1:3)';
    ys1(i,:) = Q*r_;
    rm1(:,i) = Q*rm_(:,i);
end
for i=1:4
    ii = find(is == i);
    plot3(ys1(ii,1),ys1(ii,2),ys1(ii,3),'.-')
end
%
plot3(rm1(1,:),rm1(2,:),rm1(3,:),'-','color',[.7 .7 .7])
hold off
grid on
xlabel('x (km)')
ylabel('y (km)')
zlabel('z (km)')
axis equal
legend('','S1: outbound','S2: inbound',...
'S3: Moon departure','S4: Moon approach','Moon')


function display_x(x,xf)
    x(2) = wrapTo360(x(2));
    x(3) = wrapTo360(x(3));
    x(4) = wrapTo360(x(4));
    x(8) = wrapTo360(x(8));
    x(9) = wrapTo360(x(9));
    x(10) = wrapTo360(x(10));
    x(15) = wrapTo360(x(15));
    x(16) = wrapTo360(x(16));
    x(17) = wrapTo360(x(17));
    xf(4) = wrapTo360(xf(4));
    xf(7) = wrapTo360(xf(7));
    xf(8) = wrapTo360(xf(9));
    fprintf('==================================================================\n')
    fprintf('Segment          S1            S2            S3            S4 \n')
    fprintf('------------------------------------------------------------------\n')
    fprintf('t0 (day)   %12.1f  %12.6f* %12.6f*     <= t0 S3 \n', xf(1), x(6),  x(12) )
    fprintf('dt (day)   %12.6f* %12.6f* %12.6f* %12.6f* \n', x(1),  x(7),  x(13), x(18) )
    fprintf('rp (km)    %12.1f  %12.1f  %12.1f      <= t0 S3 \n', xf(2), xf(5), xf(8) )
    fprintf('e          %12.1f  %12.1f  %12.6f*     <= t0 S3 \n', xf(3), xf(6), x(14) )
    fprintf('i  (deg)   %12.6f* %12.6f* %12.6f*     <= t0 S3 \n', x(2),  x(8),  x(15) )
    fprintf('W  (deg)   %12.6f* %12.6f* %12.6f*     <= t0 S3 \n', x(3),  x(9),  x(16) )
    fprintf('w  (deg)   %12.1f  %12.1f  %12.6f*     <= t0 S3 \n', xf(4), xf(7), x(17) )
    fprintf('nu0 (deg)  %12.6f* %12.6f* %12.1f      <= t0 S3 \n', x(4),  x(10), xf(9) )
    fprintf('dv0 (km/s) %12.6f* %12.6f*          n/a           n/a    \n', x(5), x(11) )
    fprintf('------------------------------------------------------------------\n')
end



function fobj = objective_function(x)
   fobj = x(6);
end



function [c, ceq] = constraints(x)
    global days
    % propagation
    [t1, y1, t2, y2, t3, y3, t4, y4] = propagate(x);
    %
    tf_S1 = t1(end);
    tf_S4 = t4(end);
    tf_S2 = t2(end);
    tf_S3 = t3(end);
    sv_S1 = y1(end,:);
    sv_S4 = y4(end,:);
    sv_S2 = y2(end,:);
    sv_S3 = y3(end,:);
    dt_S1 = t1(end) - t1(1);
    dt_S4 = t4(1) - t4(end);
    dt_S3 = t3(end) - t3(1);
    dt_S2 = t2(1) - t2(end);
    % inequality constraints
    c = [-dt_S1 + 1.0*days, -dt_S2 + 1.0*days, -dt_S3 + 1.0*days, -dt_S4 + 1.0*days];
    % equality constraints
    ceq = [ tf_S1 - tf_S4, sv_S1 - sv_S4, tf_S2 - tf_S3, sv_S2 - sv_S3 ];
end



function [t1, y1, t2, y2, t3, y3, t4, y4] = propagate(x)
    global et0 mu_e mu_m days xf
    % ODE integrator settings
    options = odeset('RelTol',1e-9,'AbsTol',1e-9); 
    %%% integrate S1 segment
    t0_s1 = xf(1)*days; % sec
    dt_s1 = x(1)*days; % sec
    tf_s1 = t0_s1 + dt_s1; % sec
    q_s1  = xf(2); % km
    e_s1  = xf(3);
    i_s1  = x(2); % deg
    W_s1  = x(3); % deg
    w_s1  = xf(4);  % deg
    nu0_s1 = x(4); % deg
    dv0_s1 = x(5); % km/s
    % get initial state
    [r0_, v0_] = par_to_ic([q_s1, e_s1, W_s1, i_s1, w_s1, nu0_s1],mu_e);
    % add injection delta-V
    v0_ = v0_ + dv0_s1 * v0_/norm(v0_);
    % integrate
    [t1, y1] = ode113(@(t,y) derivs(t,y),[t0_s1 tf_s1],[r0_; v0_],options);
    %%% integrate S2 segment
    t0_s2 = x(6)*days;
    dt_s2 = x(7)*days;
    tf_s2 = t0_s2 + dt_s2;
    q_s2  = xf(5); % km
    e_s2  = xf(6);
    i_s2  = x(8); % deg
    W_s2  = x(9); % deg
    w_s2  = xf(7);  % deg
    nu0_s2 = x(10); % deg
    dv0_s2 = x(11); % km/s
    % get initial state
    [r0_, v0_] = par_to_ic([q_s2, e_s2, W_s2, i_s2, w_s2, nu0_s2],mu_e);
    % add capture delta-V
    v0_ = v0_ - dv0_s2 * v0_/norm(v0_);
    % integrate
    [t2, y2] = ode113(@(t,y) derivs(t,y),[t0_s2 tf_s2],[r0_; v0_],options);
    %%% integrate S3 segment
    t0_s3 = x(12)*days;
    dt_s3 = x(13)*days;
    tf_s3 = t0_s3 + dt_s3;
    q_s3  = xf(8); % km
    e_s3  = x(14);
    i_s3  = x(15); % deg
    W_s3  = x(16); % deg
    w_s3  = x(17);  % deg
    nu0_s3 = xf(9); % deg
    % initial state, Moon-centered, J2000 frame
    [r0m_, v0m_] = par_to_ic([q_s3, e_s3, W_s3, i_s3, w_s3, nu0_s3],mu_m);
    % change to Earth-centered, J2000 frame
    [sm_, ~] = cspice_spkezr('301', et0+t0_s3, 'J2000', 'NONE', '399');
    rm_ = sm_(1:3);
    vm_ = sm_(4:6);
    % transformation is simply a translation in r_, v_
    r0_ = rm_ + r0m_;
    v0_ = vm_ + v0m_;
    % integrate
    [t3, y3] = ode113(@(t,y) derivs(t,y),[t0_s3 tf_s3],[r0_; v0_],options);
    %%% integrate S4 segment
    t0_s4 = t0_s3;
    dt_s4 = x(18)*days;
    tf_s4 = t0_s4 + dt_s4;
    % r0_ and v0_ are the same as in S3
    [t4, y4] = ode113(@(t,y) derivs(t,y),[t0_s4 tf_s4],[r0_; v0_],options);
end



function [r0_, v0_] = par_to_ic(oe,mu)
   rp  = oe(1); % periapsis distance 
   e   = oe(2); % eccentricity
   W   = oe(3); % longitude of the ascending node 
   i   = oe(4); % inclination
   w   = oe(5); % argument of periapsis 
   nu0 = oe(6); % true anomaly at epoch
   P_ = [ cosd(w)*cosd(W) - sind(w)*cosd(i)*sind(W); ...
          cosd(w)*sind(W) + sind(w)*cosd(i)*cosd(W); ...
          sind(w)*sind(i)];
   Q_ = [-sind(w)*cosd(W) - cosd(w)*cosd(i)*sind(W); ...
         -sind(w)*sind(W) + cosd(w)*cosd(i)*cosd(W); ...
          cosd(w)*sind(i)];
   p = rp*(1+e);
   r0 = p/(1 + e*cosd(nu0));
   r0_ = r0        *( cosd(nu0)*P_ +      sind(nu0 )*Q_ );
   v0_ = sqrt(mu/p)*(-sind(nu0)*P_ + (e + cosd(nu0))*Q_ );
end

function dydt = derivs(t,y)    
    global et0 mu_e mu_m days
    % all calculations in geocentric inertial (J2000) frame
    et = et0+t;
    r_ = y(1:3);
    v_ = y(4:6);
    r = norm(r_);
    [rm_, ~] = cspice_spkpos('301', et, 'J2000', 'NONE', '399');
    rm = norm(rm_);
    rms_ = rm_ - r_;
    rms = norm(rms_);
    a_ = -mu_e*r_/r^3 + mu_m*(rms_/rms^3 - rm_/rm^3);
    dydt = [v_; a_];
end
