clc;
clear all;

% Wczytanie danych CSV
data = readtable('star_classification.csv', 'VariableNamingRule', 'preserve');

% Usuniecie brakujacych danych
data = rmmissing(data);

% Wyodrebnienie cech liczbowych 
features = data{:, {'Temperature (K)','Luminosity(L/Lo)','Radius(R/Ro)','Absolute magnitude(Mv)'}};

% Konwersja etykiet do zmiennych kategorycznych. To staramy się przewidzieć
labels = categorical(data.("Star type"));

% Kodowanie zmiennych tekstowych 
StarColorEncoded = categorical(data.("Star color"));
SpectralClassEncoded = categorical(data.("Spectral Class"));

% Dodanie zakodowanych kolumn do cech liczbowych 
features = [features, double(StarColorEncoded), double(SpectralClassEncoded)];

% Normalizacja 
features = normalize(features);

% PCA Analysis
[coeff, score, ~, ~, explained] = pca(features);

% Wyświetlenie wykresu PCA
figure;
gscatter(score(:,1), score(:,2), labels);
xlabel(['Główna Składowa 1 (' num2str(explained(1), '%.2f') '% wyjaśnionej wariancji)']); 
ylabel(['Główna Składowa 2 (' num2str(explained(2), '%.2f') '% wyjaśnionej wariancji)']); 
title('Wizualizacja PCA danych gwiazd');
legend('Location', 'best');
grid on;

% Podzial danych
rng('default'); % Gwarantuje powtarzalnosc wynikow
cv = cvpartition(size(features,1), 'HoldOut', 0.3); % 30% danych na testy

XTrain = features(training(cv), :); % Cechy treningowe 
YTrain = labels(training(cv)); % Etykiety treningowe 
XTest = features(test(cv), :); % Cechy testowe
YTest = labels(test(cv)); % Etykiety testowe 

%------------Tworzenie sieci neuronowej---------------
%Liczba cech wejsciowych, 2 oznacza, że pobierana jest liczba kolumn
numFeatures = size(XTrain,2);

%Liczba klas wyjsciowcyh
numClasses = numel(categories(YTrain));

layers = [
    featureInputLayer(numFeatures, 'Normalization','none','Name','Input')
    fullyConnectedLayer(25,"Name",'Hiden') %Warstwa ukryta z 25 neuronami
    reluLayer('Name', 'ReLU'); %Funkcja aktywacji
    fullyConnectedLayer(numClasses,"Name",'Output'); %Warstwa wyjsciowa
    softmaxLayer("Name",'Softmax') %Normalizacja do prawdopobieńśtwa
    classificationLayer('Name','ClassOutput') %Warstwa klasyfikacyjna
    ];

% Ustawienie opcji treningu
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 12, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Trening sieci
net = trainNetwork(XTrain, YTrain, layers, options);

% Testowanie sieci
YPredNN = classify(net, XTest);

% Obliczenie dokładności
accuracyNN = mean(YPredNN == YTest);
disp("Dokładność sieci neuronowej: " + accuracyNN);

%-----------Tworzenie modelu Random Forest------------
randomForestModel = fitcensemble(XTrain, YTrain, "Method", "Bag", "NumLearningCycles", 100, 'Learners', 'tree');

% Predykcja na zbiorze testowym 
YPredRF = predict(randomForestModel, XTest);

% Obliczenie dokladnosci 
accuracyRF = mean(YPredRF == YTest);
disp("Dokladnosc modelu Random Forest: " + accuracyRF);

% -----------------------------SVM--------------------------
svmModel = fitcecoc(XTrain, YTrain, "Coding", "onevsall", "Learners", templateSVM('KernelFunction', 'linear', 'Standardize', true));

% Predykcja na zbiorze testowym dla SVM
YPredSVM = predict(svmModel, XTest);

% Obliczenie dokladnosci dla SVM 
accuracySVM = mean(YPredSVM == YTest);
disp("Dokładność modelu SVM: " + accuracySVM);

% ----------------------------KNN---------------------------
k = 5; % Liczba sąsiadów
knnModel = fitcknn(XTrain, YTrain, 'NumNeighbors', k, 'Standardize', true);
YPredKNN = predict(knnModel, XTest);
accuracyKNN = mean(YPredKNN == YTest);
disp("Dokładność KNN: " + accuracyKNN);

% Macierz konfuzji dla wszystkich metod
figure;
subplot(1,4,1);
confusionchart(YTest, YPredRF);
title('Random Forest');

subplot(1,4,2);
confusionchart(YTest, YPredSVM);
title('SVM');

subplot(1,4,3);
confusionchart(YTest, YPredKNN);
title('KNN');

subplot(1,4,4);
confusionchart(YTest, YPredNN);
title('Sieć neuronowa');

% ------------------Obliczanie znaczenia cech-----------------
% Wyświetlenie znaczenia cech dla Random Forest
featureImportance = oobPermutedPredictorImportance(randomForestModel);
figure;
bar(featureImportance);
title('Znaczenie cech dla Random Forest');
xlabel('Cechy');
ylabel('Znaczenie');
xticklabels({'Temperature', 'Luminosity', 'Radius', 'Magnitude', 'Star Color', 'Spectral Class'});
