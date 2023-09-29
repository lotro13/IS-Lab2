% # IS-Lab2 (LT)
% % Intelektualiosios sistemos. Antrojo laboratorinio darbo užduotis.
% # Tikslas
% Išmokti savarankiškai suprogramuoti paprasto netiesinio aproksimatoriaus mokymo (parametrų skaičiavimo) algoritmą.
% # Užduotys (maks. 8 balai)
% 1. Sukurkite daugiasluoksnio perceptrono koeficientams apskaičiuoti skirtą programą. Daugiasluoksnis perceptronas turi atlikti aproksimatoriaus funkciją. Daugiasluoksnio perceptrono struktūra:
% - vienas įėjimas (įėjime paduodamas 20 skaičių vektorius X, su reikšmėmis intervale nuo 0 iki 1, pvz., x = 0.1:1/22:1; ).
% - vienas išėjimas (pvz., išėjime tikimasi tokio norimo atsako, kurį galima būtų apskaičiuoti pagal formulę: y = (1 + 0.6\*sin(2\*pi\*x/0.7)) + 0.3\*sin(2\*pi\*x))/2; - kuriamas neuronų tinklas turėtų "modeliuoti/imituoti šios formulės elgesį" naudodamas visiškai kitokią matematinę išraišką nei ši);
% - vienas paslėptasis sluoksnis su hiperbolinio tangento arba sigmoidinėmis aktyvavimo funkcijomis neuronuose (neuronų skaičius: 4-8);
% - tiesine aktyvavimo funkcija išėjimo neurone;
% - mokymo algoritmas - Backpropagation (atgalinio sklidimo).
% # Papildoma užduotis (papildomi 2 balai)
% Išspręskite paviršiaus aproksimavimo uždavinį, kai tinklas turi du įėjimus ir vieną išėjimą.
% # Rekomenduojama literatūra
% - Neural Networks and Learning Machines (3rd Edition), <...> psl., <...> lentelė
close all
clear all
% Duomenu pruosimas
x1 = 0.1:0.05:1;
x2 = 0.1:0.05:1;
x3 = 0.1:0.01:1;
x4 = 0.1:0.01:1;
d1 = (1 + 0.6*sin(2*pi*x1/0.7)) + (0.3*sin(2*pi*x1))/2;
d2 = 0.12*(sin(x1*2*pi/12) + cos(x2*2*pi/2)) + 0.05;
plot(1:19,d1,'kx',1:19,d2,'rx')
% Tinklo struktura
% 2 iejimai; 4 paslepti; 2 isejimai

% 3 Pradiniu reiksmiu pasirinkimas
% I. sluoksnis
w11_1 = randn(1); w12_1 = randn(1); b1_1 = randn(1);
w21_1 = randn(1); w22_1 = randn(1); b2_1 = randn(1);
w31_1 = randn(1); w32_1 = randn(1); b3_1 = randn(1);
w41_1 = randn(1); w42_1 = randn(1); b4_1 = randn(1);
% II. sluoksnis
w11_2 = randn(1); w12_2 = randn(1); w13_2 = randn(1); w14_2 = randn(1); b1_2 = randn(1);
w21_2 = randn(1); w22_2 = randn(1); w23_2 = randn(1); w24_2 = randn(1); b2_2 = randn(1);
n = 0.1; % mokymo zyngsnis

% 4. Tinklo atsako skaiciavimas y(v) = 1/(1 + exp(-v)) - aktyvavimo func.
% y|v = (v * (1 - v))
for epoch = 1:1000
    for indx = 1:19
        % I. sluoksnis
        v1_1 = x1(indx)*w11_1 + x2(indx)*w12_1 + b1_1; y1_1 = 1/(1 + exp(-v1_1));
        v2_1 = x1(indx)*w21_1 + x2(indx)*w22_1 + b2_1; y2_1 = 1/(1 + exp(-v2_1));
        v3_1 = x1(indx)*w31_1 + x2(indx)*w32_1 + b3_1; y3_1 = 1/(1 + exp(-v3_1));
        v4_1 = x1(indx)*w41_1 + x2(indx)*w42_1 + b4_1; y4_1 = 1/(1 + exp(-v4_1));
        % II. sluoksnis
        v1_2 = y1_1*w11_2 + y2_1*w12_2 + y3_1*w13_2 + y4_1*w14_2 + b1_2; y1 = v1_2;
        v2_2 = y1_1*w21_2 + y2_1*w22_2 + y3_1*w23_2 + y4_1*w24_2 + b2_2; y2 = v2_2;

        % 5. Klaidos skaiciavimas E = 1/2e1^2 + 1/2*e2^2 -tikslo funkcija E|e1 = e1
        e1 = d1(indx) - y1;
        e2 = d2(indx) - y2;

        % 6. Tinklo koeficientu atnaujinimas
        % w = w + n*delta*IN
        % delta_out = aktuvavimo_funkcijos_isvestine|v * tikslo_funkcijos_isvestine|e
        % delta_hidden = aktuvavimo_funkcijos_isvestine|v * (w1*delta_out1 + w2*delta_out2 + ..)
        % Klaidos gradijentai
        delta_out1 = 1 * 1/2*2*e1^1; % y(v)|v * E|e1
        delta_out2 = 1 * 1/2*2*e2^1; % y(v)|v * E|e1

        delta_hidden1 = y1_1*(1-y1_1) * (w11_2*delta_out1 + w21_2*delta_out2);
        delta_hidden2 = y2_1*(1-y2_1) * (w12_2*delta_out1 + w22_2*delta_out2);
        delta_hidden3 = y3_1*(1-y3_1) * (w13_2*delta_out1 + w23_2*delta_out2);
        delta_hidden4 = y4_1*(1-y4_1) * (w14_2*delta_out1 + w24_2*delta_out2);

        w11_2 = w11_2 + n*delta_out1*y1_1; w12_2 = w12_2 + n*delta_out1*y2_1;
        w13_2 = w13_2 + n*delta_out1*y3_1; w14_2 = w14_2 + n*delta_out1*y4_1;
        b1_2 = b1_2 + n*delta_out1*1;

        w21_2 = w21_2 + n*delta_out2*y1_1; w22_2 = w22_2 + n*delta_out2*y2_1;
        w23_2 = w23_2 + n*delta_out2*y3_1; w24_2 = w24_2 + n*delta_out2*y4_1;
        b2_2 = b2_2 + n*delta_out2*1;

        w11_1 = w11_1 + n*delta_hidden1*x1(indx); w12_1 = w12_1 + n*delta_hidden1*x2(indx);
        b1_1 = b1_1 + n*delta_hidden1;
        w21_1 = w21_1 + n*delta_hidden2*x1(indx); w22_1 = w22_1 + n*delta_hidden2*x2(indx);
        b2_1 = b2_1 + n*delta_hidden2;
        w31_1 = w31_1 + n*delta_hidden3*x1(indx); w32_1 = w32_1 + n*delta_hidden3*x2(indx);
        b3_1 = b3_1 + n*delta_hidden3;
        w41_1 = w41_1 + n*delta_hidden4*x1(indx); w42_1 = w42_1 + n*delta_hidden4*x2(indx);
        b4_1 = b4_1 + n*delta_hidden4;
    end

    for indx = 1:91
        % I. sluoksnis
        v1_1 = x3(indx)*w11_1 + x4(indx)*w12_1 + b1_1; y1_1 = 1/(1 + exp(-v1_1));
        v2_1 = x3(indx)*w21_1 + x4(indx)*w22_1 + b2_1; y2_1 = 1/(1 + exp(-v2_1));
        v3_1 = x3(indx)*w31_1 + x4(indx)*w32_1 + b3_1; y3_1 = 1/(1 + exp(-v3_1));
        v4_1 = x3(indx)*w41_1 + x4(indx)*w42_1 + b4_1; y4_1 = 1/(1 + exp(-v4_1));
        % II. sluoksnis
        v1_2 = y1_1*w11_2 + y2_1*w12_2 + y3_1*w13_2 + y4_1*w14_2 + b1_2; y1 = v1_2;
        v2_2 = y1_1*w21_2 + y2_1*w22_2 + y3_1*w23_2 + y4_1*w24_2 + b2_2; y2 = v2_2;

        Y1(indx) = y1;
        Y2(indx) = y2;
    end
end
figure(2)
plot(1:91, Y1, 'ko', 1:91, Y2, 'ro')
