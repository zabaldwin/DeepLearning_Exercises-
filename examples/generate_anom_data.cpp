#include <iostream>                                                                                                                                                                                   
#include <vector>
#include <TRandom3.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TCanvas.h>

using namespace std;

// Function to generate normal data
vector<double> generateNormalData(int numEvents) {
    TRandom3 randGen;
    vector<double> data;
    for (int i = 0; i < numEvents; ++i) {
        // Simulate normal response (example: Gaussian distribution)
        double response = randGen.Gaus(0, 1);
        data.push_back(response);
    }
    return data;
}

// Function to inject anomalies into normal data
void injectAnomalies(vector<double>& data, int numAnomalies) {
    TRandom3 randGen;
    int dataSize = data.size();
    for (int i = 0; i < numAnomalies; ++i) {
        // Randomly select an index to inject anomaly
        int index = randGen.Integer(dataSize);
        // Simulate anomaly (example: set value to a large number)
        data[index] = randGen.Uniform(5, 10);
    }
}

int main() {
    // Number of normal events to generate
    int numNormalEvents = 100000;
    // Number of anomalies to inject
    int numAnomalies = 2000;

    // Generate normal data
    vector<double> normalData = generateNormalData(numNormalEvents);

    // Inject anomalies into normal data
    injectAnomalies(normalData, numAnomalies);

    // Create a ROOT file to store the mixed dataset
    TFile outputFile("mixedData.root", "RECREATE");

    // Create a TTree
    TTree tree("tree", "Particle Detector Responses");

    // Create a branch to store the detector responses
    double response;
    tree.Branch("response", &response);

    // Fill the tree with mixed data (only after injecting anomalies)
    for (const auto& dataPoint : normalData) {
        response = dataPoint;
        tree.Fill();
    }

    // Write the tree to the ROOT file
    tree.Write();
    outputFile.Close();

    // Create a histogram to visualize the response distribution after injecting anomalies
    TH1D histAfterInjection("response_histogram_after_injection", "Response Distribution (After Injection)", 100, -10, 10);
    for (const auto& dataPoint : normalData) {
        histAfterInjection.Fill(dataPoint);
    }

    TCanvas canvasAfter("canvas_after_injection", "Response Histogram (After Injection)", 800, 600);
    histAfterInjection.Draw();
    histAfterInjection.GetXaxis()->SetTitle("Response");
    histAfterInjection.GetYaxis()->SetTitle("Frequency");
    canvasAfter.Draw();
    canvasAfter.SaveAs("response_histogram_after_injection.png");

    return 0;
}
