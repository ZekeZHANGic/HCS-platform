blockSize = 50;
inputDir = getDirectory("Choose the directory containing the images");
outputDir = getDirectory("Choose the directory to save the processed images");
list = getFileList(inputDir);
setBatchMode(true);

for (i = 0; i < list.length; i++) {
    if (endsWith(list[i], ".tif") || endsWith(list[i], ".tiff")) {
        open(inputDir + list[i]);
        originalImage = getImageID();
        originalTitle = getTitle();
        width = getWidth();
        height = getHeight();
        run("Duplicate...", "title=Background");
        backgroundImage = getImageID();
        setColor(0);
        makeRectangle(0, 0, width, height);
        run("Clear", "slice");
        selectImage(originalImage);
        run("Duplicate...", "title=Background2");
        backgroundImage2 = getImageID();
        setColor(0);
        makeRectangle(0, 0, width, height);
        run("Clear", "slice");
        selectImage(originalImage);
        for (x = 0; x < width; x += blockSize) {
            for (y = 0; y < height; y += blockSize) {
                makeRectangle(x, y, blockSize, blockSize);
                run("Measure");
                median = getValue("Median");
                print("Block at (" + x + "," + y + ") - Median: " + median);
                selectImage(backgroundImage);
                setColor(median);
                makeRectangle(x, y, blockSize, blockSize);
                run("Fill", "slice");
                selectImage(originalImage);
            }
        }
        selectImage(backgroundImage);
        for (x = 0; x < width; x += 400) {
            for (y = 0; y < height; y += 50) {
                makeRectangle(x, y, 400, 50);
       
                run("Measure");
                mode = getValue("Median");
                print("Block at (" + x + "," + y + ") - Mode: " + mode);
                selectImage(backgroundImage2);
                setColor(mode);
                makeRectangle(x, y, 400, 50);
                run("Fill", "slice");
                selectImage(backgroundImage);
            }
        }

      
        selectImage(originalImage);
        imageCalculator("Subtract create", originalTitle, "Background2");
        saveAs("Tiff", outputDir + "" + list[i]);
        close("*");
    }
}

setBatchMode(false);


blockSizes = newArray(50, 100, 200);
inputDir = getDirectory("Choose the directory containing the images");
outputDir = getDirectory("Choose the directory to save the processed images");
list = getFileList(inputDir);
setBatchMode(true);

for (i = 0; i < list.length; i++) {
    if (endsWith(list[i], ".tif") || endsWith(list[i], ".tiff")) {
        open(inputDir + list[i]);
        originalImage = getImageID();
        originalTitle = getTitle();
        width = getWidth();
        height = getHeight();

        for (x = 0; x < width; x += 50) {
            for (y = 0; y < height; y += 50) {
                makeRectangle(x, y, 50, 50);
                run("Measure");
                localVariance = getValue("StdDev");

                // 根据局部方差选择块大小
                blockSize = getAdaptiveBlockSize(x, y, localVariance);

                // 处理图像分块
                makeRectangle(x, y, blockSize, blockSize);
                run("Measure");
                median = getValue("Median");
                print("Block at (" + x + "," + y + ") - Median: " + median);
                setColor(median);
                run("Fill", "slice");
            }
        }

        saveAs("Tiff", outputDir + "" + list[i]);
        close("*");
    }
}

setBatchMode(false);

function getAdaptiveBlockSize(x, y, localVariance) {
    if (localVariance < 15) {
        return 200;
    } else if (localVariance > 50) {
        return 50;
    } else {
        return 100;
    }
}
