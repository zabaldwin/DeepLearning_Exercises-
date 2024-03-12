#include <iostream>                                                                                                                                                                                   
#include <vector>
#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TRandom3.h>

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

int main() {
    // Number of normal events to generate
    int numNormalEvents = 100000;

    // Generate normal data
    vector<double> normalData = generateNormalData(numNormalEvents);

    // Create a ROOT file to store the data
    TFile outputFile("normalData.root", "RECREATE");

    // Create a TTree
    TTree tree("tree", "Particle Detector Responses");

    // Create a branch to store the detector responses
    double response;
    tree.Branch("response", &response);

    // Fill the tree with normal data
    for (const auto& dataPoint : normalData) {
        response = dataPoint;
        tree.Fill();
    }

    // Write the tree to the ROOT file
    tree.Write();
    outputFile.Close();

    cout << "Normal data generation completed. Saved to normalData.root." << endl;

    // Create a histogram to visualize the response distribution
    TH1D histNormal("response_histogram_", "Response Distribution", 100, -10, 10);
    for (const auto& dataPoint : normalData) {
        histNormal.Fill(dataPoint);
    }

    TCanvas canvas("response_canvas_normal", "Response Histogram (Normal)", 800, 600);
    histNormal.Draw();
    canvas.SaveAs("response_histogram_normal.png");

    return 0;
}
