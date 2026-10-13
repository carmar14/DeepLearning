
%% GENERACION DE REGISTROS CELULARES CON SIMULINK
clear; clc;
close all

model = 'modelo_data';
load_system(model);

% Tiempo de simulacion
tFinal = 100;
set_param(model, 'StopTime', num2str(tFinal));

% Parametros del modelo biologico
p.r     = 1.0;
p.K     = 1.0;
p.alpha = 0.8;
p.s     = 0.1;
p.rho   = 0.5;
p.h     = 0.3;
p.delta = 0.2;
p.eta   = 0.1;

% Escenarios de condiciones iniciales
T0_values = linspace(0.1, 0.9, 6);
E0_values = linspace(0.05, 0.65, 20);

% Tabla acumulada de todas las simulaciones
Dataset = table();

scenario = 0;

for i = 1:length(T0_values)
    for j = 1:length(E0_values)

        scenario = scenario + 1;

        T0 = T0_values(i);
        E0 = E0_values(j);

        % Configurar esta simulacion
        simIn = Simulink.SimulationInput(model);

        simIn = simIn.setVariable('T0', T0);
        simIn = simIn.setVariable('E0', E0);

        % Parametros biologicos
        names = fieldnames(p);
        for k = 1:numel(names)
            simIn = simIn.setVariable( ...
                names{k}, p.(names{k}));
        end

        % Ejecutar simulacion
        simOut = sim(simIn);

        % Recuperar series temporales
        Tsig = simOut.get('T_out');
        Esig = simOut.get('E_out');

        t = Tsig.Time(:);
        T = squeeze(Tsig.Data);
        E = squeeze(Esig.Data);

        T = T(:);
        E = E(:);

        % Verificar consistencia de las señales
        assert(numel(t) == numel(T) && ...
               numel(t) == numel(E), ...
               'Las senales tienen longitudes diferentes.');

        % Construir registro
        n = numel(t);

        D = table( ...
            repmat(scenario,n,1), ...
            t, T, E, ...
            repmat(T0,n,1), ...
            repmat(E0,n,1), ...
            'VariableNames', ...
            {'Scenario','Time','Tumor','Effector', ...
             'T0','E0'});

        % Agregar parametros verdaderos
        for k = 1:numel(names)
            D.(names{k}) = ...
                repmat(p.(names{k}),n,1);
        end

        Dataset = [Dataset; D];

        fprintf('Escenario %d completado: T0=%.3f, E0=%.3f\n', ...
                scenario, T0, E0);
    end
end

% Exportar los registros
writetable(Dataset, 'CellularDataset.csv');
save('CellularDataset.mat', 'Dataset', 'p');

fprintf('\nTotal de escenarios: %d\n', scenario);
fprintf('Total de muestras: %d\n', height(Dataset));

% Visualizar las trayectorias
figure;
tiledlayout(2,1);

nexttile;
hold on;
for s = 1:scenario
    idx = Dataset.Scenario == s;
    plot(Dataset.Time(idx), Dataset.Tumor(idx));
end
ylabel('Poblacion tumoral');
grid on;

nexttile;
hold on;
for s = 1:scenario
    idx = Dataset.Scenario == s;
    plot(Dataset.Time(idx), Dataset.Effector(idx));
end
xlabel('Tiempo');
ylabel('Poblacion efectora');
grid on;
