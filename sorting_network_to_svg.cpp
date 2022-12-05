#include "sorting_network.h"

using std::pair;
using std::vector;

void print_sorting_network_as_svg(const char *filename, int n, const vector<pair<int, int>> &network, int scale) {
    printf("%s %d\n", filename, (int)network.size());
    // Figure out the x coordinate for each link
    struct Link {
        int a, b, x;
    };
    vector<Link> links;
    {
        for (auto it = network.begin(); it != network.end(); it++) {
            const auto &p = *it;
            int x = 0;
            for (auto &l : links) {
                if ((l.b >= p.first &&
                     l.a <= p.second)) {
                    x = std::max(x, l.x + 1);
                }
            }
            links.emplace_back(Link{p.first, p.second, x});
        }
    }

    FILE *f = fopen(filename, "w");
    const int left = 0;
    const int right = (links.back().x + 3) * scale;
    const int top = 0;
    const int bottom = (n + 1) * scale;

    const char *header = R"header(<?xml version="1.0" encoding="utf-8"?>
<!-- Generator: Adobe Illustrator 25.0.1, SVG Export Plug-In . SVG Version: 6.00 Build 0)  -->
<svg version="1.2" baseProfile="tiny" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
         x="0px" y="0px" viewBox="0 0 %d %d" overflow="visible" xml:space="preserve">
)header";

    fprintf(f, header, right, bottom);

    // Draw the wires
    for (int i = 0; i < n; i++) {
        fprintf(f, "<line stroke=\"#a0a0a0\" stroke-width=\"1\" x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\"/>\n",
                left, (i + 1) * scale, right, (i + 1) * scale);
    }

    // Draw the links
    for (auto &l : links) {
        const int y0 = (l.a + 1) * scale;
        const int y1 = (l.b + 1) * scale;
        const int x = (l.x + 1) * scale;
        fprintf(f, "<g>\n"
                   "    <line stroke=\"#000000\" stroke-width=\"1\" x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\"/>\n"
                   "    <circle cx=\"%d\" cy=\"%d\" r=\"2\"/>\n"
                   "    <circle cx=\"%d\" cy=\"%d\" r=\"2\"/>\n"
                   "</g>\n",
                x, y0, x, y1,
                x, y0, x, y1);
    }

    fprintf(f, "</svg>\n");
    fclose(f);
}

int main(int argc, char **argv) {
    print_sorting_network_as_svg("median_27.svg", 27, even_odd_sort(27, 13, 13, false), 6);
    print_sorting_network_as_svg("median_7_pairs_already_sorted.svg", 7, even_odd_sort(7, 3, 3, true), 24);
    return 0;
}
