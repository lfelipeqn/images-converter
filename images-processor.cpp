#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
namespace fsys = std::filesystem;

void processImage(const string& path, int canvasWidth, int canvasHeight) {
    // 1) Read the image
    Mat image = imread(path);
    if (image.empty()) {
        cerr << "Could not read image: " << path << endl;
        return;
    }

    // 2) Calculate scaling factor using the COVER strategy:
    // Use max(scaleX, scaleY) so that the image fills the entire canvas.
    double scaleX = static_cast<double>(canvasWidth) / image.cols;
    double scaleY = static_cast<double>(canvasHeight) / image.rows;
    double scale  = max(scaleX, scaleY);

    int newWidth  = static_cast<int>(image.cols * scale);
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
    int cropX = (newWidth  - canvasWidth)  / 2;
    int cropY = (newHeight - canvasHeight) / 2;
    Rect cropRect(cropX, cropY, canvasWidth, canvasHeight);
    Mat canvas = bgra(cropRect).clone();

    // 8) Generate output file paths.
    fsys::path p(path);
    string base = p.stem().string();
    fsys::path dir = p.parent_path();
    fsys::path pngPath  = dir / (base + "_processed.png");
    fsys::path webpPath = dir / (base + "_processed.webp");

    // 9) Save the processed image as PNG and WebP.
    imwrite(pngPath.string(), canvas);
    imwrite(webpPath.string(), canvas);

    // 10) (Optional) Set file permissions (read/write for everyone)
    using fsys::perms;
    using fsys::perm_options;
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

    cout << "Processed: " << path << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <folder> <canvasWidth> <canvasHeight>" << endl;
        return 1;
    }

    string folder = argv[1];
    int canvasWidth  = atoi(argv[2]);
    int canvasHeight = atoi(argv[3]);

    // Supported image extensions.
    vector<string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};

    // Process each supported file in the folder.
    for (const auto& entry : fsys::directory_iterator(folder)) {
        if (!entry.is_regular_file())
            continue;

        string ext = entry.path().extension().string();
        transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
            processImage(entry.path().string(), canvasWidth, canvasHeight);
        }
    }

    return 0;
}
