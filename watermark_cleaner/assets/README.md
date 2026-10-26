# Local logo silhouettes

`tiktok.svg` and `instagram.svg` are unmodified Simple Icons 16.0.0 assets:

- https://github.com/simple-icons/simple-icons/blob/16.0.0/icons/tiktok.svg
- https://github.com/simple-icons/simple-icons/blob/16.0.0/icons/instagram.svg

Their PNG files are 96 by 96 alpha-channel rasterizations made with resvg-py 0.5.0. The upstream CC0-1.0 dedication is retained in `THIRD_PARTY_LICENSES/SimpleIcons-CC0.txt`.

`reels.svg` is an authored geometric clapperboard/play silhouette. Its PNG is rasterized the same way. These two files use the app's Apache-2.0 license. It is a matching aid for this icon layout, not a complete catalog of Instagram export versions.

The app loads these tiny local PNGs with Pillow. It does not download icons, execute SVG, or require a rasterizer at runtime. Brand names remain their owners' trademarks.

TikTok and Instagram use silhouette matching. CapCut and InVideo profiles currently constrain named-wordmark and adjoining creator masks. Unsupported symbols still require visual correction.
