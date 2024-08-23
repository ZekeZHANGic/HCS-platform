// ImageJ Macro for Rolling Ball Algorithm with Median Intensity in Batch Processing

// 1. Choose the directory containing the images
inputDir = getDirectory("Choose the Input Directory");

// 2. Choose the directory to save the processed images
outputDir = getDirectory("Choose the Output Directory");

// Get list of all files in the directory
list = getFileList(inputDir);

// Loop through each file in the directory
for (i = 0; i < list.length; i++) {
    filename = list[i];

    // Check if the file name contains "brightfield"
    if (indexOf(filename, "Brightfield") != -1) {
        // Open the image
        open(inputDir + filename);

        // Get image statistics
        getStatistics(area, mean, min, max, std, histogram);

        // Calculate min and max for brightness/contrast adjustment
        newMin = mean - 2 * std;
        newMax = mean + 2 * std;

        // Adjust the pixel values
       
        setMinAndMax(newMin, newMax);
        run("Apply LUT"); // Apply the lookup table to adjust pixel values directly


        // Save the adjusted image
        saveAs("Tiff", outputDir + "adjusted_" + filename);

        // Close the image
        close();
    }
}
