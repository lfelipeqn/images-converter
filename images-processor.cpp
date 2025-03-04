#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
namespace fsys = std::filesystem;

// Structure to define image sizes and labels
struct ImageSize {
    int width;
    int height;
    string label;
};

void processImage(const string& path, const vector<ImageSize>& sizes) {
    // 1) Read the image
    Mat image = imread(path);
    if (image.empty()) {
        cerr << "Could not read image: " << path << endl;
        return;
    }

    fsys::path p(path);
    string base = p.stem().string();
    fsys::path dir = p.parent_path();
    fsys::path outputDir = dir / base; // Create a folder named after the original file

    // Create the output directory if it doesn't exist
    if (!fsys::exists(outputDir)) {
        fsys::create_directory(outputDir);
        // Set folder permissions to public write
        fsys::permissions(
            outputDir,
            fsys::perms::owner_read | fsys::perms::owner_write | fsys::perms::owner_exec |
            fsys::perms::group_read | fsys::perms::group_write | fsys::perms::group_exec |
            fsys::perms::others_read | fsys::perms::others_write | fsys::perms::others_exec,
            fsys::perm_options::replace
        );
    }
    
    //Save original in webp and png
    fsys::path originalPngPath = outputDir / (base + ".png");
    fsys::path originalWebpPath = outputDir / (base + ".webp");
    imwrite(originalPngPath.string(), image);
    imwrite(originalWebpPath.string(), image);
    
    //Set file permissions to public write
    using fsys::perms;
    using fsys::perm_options;
    fsys::permissions(
        originalPngPath,
        perms::owner_read | perms::owner_write |
        perms::group_read | perms::group_write |
        perms::others_read | perms::others_write,
        perm_options::replace
    );
    fsys::permissions(
        originalWebpPath,
        perms::owner_read | perms::owner_write |
        perms::group_read | perms::group_write |
        perms::others_read | perms::others_write,
        perm_options::replace
    );
    

    // Iterate through each size and generate resized images
    for (const auto& size : sizes) {
        int canvasWidth = size.width;
        int canvasHeight = size.height;

        // 2) Calculate scaling factor using the COVER strategy:
        // Use max(scaleX, scaleY) so that the image fills the entire canvas.
        double scaleX = static_cast<double>(canvasWidth) / image.cols;
        double scaleY = static_cast<double>(canvasHeight) / image.rows;
        double scale = max(scaleX, scaleY);

        int newWidth = static_cast<int>(image.cols * scale);
        int newHeight = static_cast<int>(image.rows * scale);

        // 3) Resize the image to the new dimensions.
        Mat resized;
        resize(image, resized, Size(newWidth, newHeight), 0, 0, INTER_LINEAR);

        // 4) Create a mask for background removal via flood fill.
        // Note: The mask must be 2 pixels larger than the image.
        Mat mask = Mat::zeros(resized.rows + 2, resized.cols + 2, CV_8UC1);

        // Adjust tolerance as needed.
        Scalar tolerance(10, 10, 10);

        // Flood fill from all four corners.
        vector<Point> seeds = {
            Point(0, 0),
            Point(resized.cols - 1, 0),
            Point(0, resized.rows - 1),
            Point(resized.cols - 1, resized.rows - 1)
        };
        for (const auto& seed : seeds) {
            floodFill(resized, mask, seed, Scalar(0, 0, 0), nullptr, tolerance, tolerance,
                      8 | FLOODFILL_MASK_ONLY);
        }

        // 5) Convert the resized image to BGRA for transparency.
        Mat bgra;
        cvtColor(resized, bgra, COLOR_BGR2BGRA);

        // 6) Set the alpha channel based on the flood fill mask.
        //    If a pixel in the mask (offset by 1) is non-zero, set alpha to 0 (transparent).
        for (int y = 0; y < bgra.rows; y++) {
            for (int x = 0; x < bgra.cols; x++) {
                if (mask.at<uchar>(y + 1, x + 1) != 0)
                    bgra.at<Vec4b>(y, x)[3] = 0;   // Transparent
                else
                    bgra.at<Vec4b>(y, x)[3] = 255; // Opaque
            }
        }

        // 7) Crop the centered region to exactly canvasWidth x canvasHeight.
        // Since we used the COVER strategy, one dimension will be equal or larger than the canvas.
        int cropX = (newWidth - canvasWidth) / 2;
        int cropY = (newHeight - canvasHeight) / 2;
        Rect cropRect(cropX, cropY, canvasWidth, canvasHeight);
        Mat canvas = bgra(cropRect).clone();

        // 8) Generate output file paths.
        fsys::path pngPath = outputDir / (size.label + "_" + base + ".png");
        fsys::path webpPath = outputDir / (size.label + "_" + base + ".webp");

        // 9) Save the processed image as PNG and WebP.
        imwrite(pngPath.string(), canvas);
        imwrite(webpPath.string(), canvas);

        // 10) Set file permissions to public write
        fsys::permissions(
            pngPath,
            perms::owner_read | perms::owner_write |
            perms::group_read | perms::group_write |
            perms::others_read | perms::others_write,
            perm_options::replace
        );
        fsys::permissions(
            webpPath,
            perms::owner_read | perms::owner_write |
            perms::group_read | perms::group_write |
            perms::others_read | perms::others_write,
            perm_options::replace
        );
    }

    cout << "Processed: " << path << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <folder>" << endl;
        return 1;
    }

    string folder = argv[1];

    // Define the image sizes and labels
    vector<ImageSize> sizes = {
        {120, 120, "xs"},
        {300, 300, "sm"},
        {600, 600, "md"},
        {800, 800, "lg"}
    };

    // Supported image extensions.
    vector<string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};

    // Process each supported file in the folder.
    for (const auto& entry : fsys::directory_iterator(folder)) {
        if (!entry.is_regular_file())
            continue;

        string ext = entry.path().extension().string();
        transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
            processImage(entry.path().string(), sizes);
        }
    }

    return 0;
}
