# Vendored upstream: PufferLib Ocean NMMO3

- **Repo:** https://github.com/PufferAI/PufferLib
- **Commit:** `c5d3c637446047a6efbcaa74c039c5295d201ab0` (branch 4.0)
- **License:** MIT (see `vendor/LICENSE-pufferlib`); render assets carry their
  own license (`resources/nmmo3/ASSETS_LICENSE.md`, vendored)
- **Fetched:** 2026-08-02

## Rule: files under `vendor/upstream/` are byte-pristine

Never edit files in `vendor/upstream/`. All modifications are patch files in
`sim/patches/`, applied at build time into `build/` by `sim/apply_patches.sh`.
Pristineness is verifiable by diffing against the pinned upstream commit and
by the sha256 sums below.

## Source files

| vendored file | upstream path | sha256 |
|---|---|---|
| `nmmo3.h` | `ocean/nmmo3/nmmo3.h` | `42569fb367cd38fe8f8beedf31082f7875c98b9deae85bf17842d96a1c8321af` |
| `nmmo3.c` | `ocean/nmmo3/nmmo3.c` | `7ceed09979eebb2aef1c7ee508644ca615bc59243eb2e19799801b0fc143730a` |
| `binding.c` | `ocean/nmmo3/binding.c` | `628f55c801b93b475a43b8f3b3716de3b14330d7afd6b3630b3b9b75f6c35ab8` |
| `simplex.h` | `ocean/nmmo3/simplex.h` | `3551b1516403ff36f807557c1deb10f654f249c3f4967b45d924c9c0b0f8892d` |
| `tile_atlas.h` | `ocean/nmmo3/tile_atlas.h` | `2a5eafc813d9a40cf56c420fb60a7e8d108875c7b208d6e19a3b6091cc4b05a8` |
| `nmmo3.ini` | `config/nmmo3.ini` | `f9a61d66b23aab1f2c66abef318e9dcd1789f50ffb4bd92d4c3c33b88a0e7790` |
| `puffernet.h` | `src/puffernet.h` | `f7f53ca1a1d1a56190bc8c73a099d5ac356013da3e4abdb0050342e33b88405b` |

`puffernet.h` is upstream's dependency-free C inference library; the baseline
player (Phase N3) compiles it with `nmmo3_weights.bin` into the brain wasm,
mirroring the `MMONet` construction in the vendored `nmmo3.c` demo.

## Render assets (`vendor/upstream/resources/nmmo3/`)

Enumerated from the renderer's `make_client` (`nmmo3.h:2456-2531`): the map
shader (both GLSL variants; 100 = web, 330 = desktop), the merged tile sheet,
the item sheet, three inventory-slot PNGs, and 50 character sheets
(5 elements x 10 sheets). Plus the pretrained policy weights
(`nmmo3_weights.bin`, 17,723,904 bytes = 4,430,976 float32 params) and the
upstream assets license.

| vendored file | sha256 |
|---|---|
| `ASSETS_LICENSE.md` | `f2316bd9cf9ccca4805e4a4f85df619ad516db97ae0fe4aa1538a58072dccc67` |
| `air_0.png` | `814bee9ae695a1ad36d649e85b4c77e50ef9e135e783538a33046144edfa900f` |
| `air_1.png` | `bc392fd1e317b57f596ed1136c8c32d290a730ca78118970c9deaa13f1ed94d7` |
| `air_2.png` | `e428e0b68cd02e2f16c9a566b109db99c920bac57439121d8fa3aa281e0dfda2` |
| `air_3.png` | `e7bf585c69f1c78bd54c7830d3f7977db032ff872eecf815b1da0de5cf3a143e` |
| `air_4.png` | `adda2e584b913eb9405b575808e41533479ac86f9b4d810b5db17ab932822d33` |
| `air_5.png` | `25eda34d0df63cc7c5237e4cf92cdca17ebebf7d4665cfbee7fbf768dddf4bb0` |
| `air_6.png` | `01a6b5e84e9c0d3a3edcab7a896d0822be36e364d8347decde7f9c98ab34e6ec` |
| `air_7.png` | `8d5eee81a61b4dc197e53f8fc014675fbff1b4e027ab0af5d4b0ef3fffab5fe9` |
| `air_8.png` | `1505a6ece301283f8f0021ac2662e88f6bed817696c871bb08a14ea115904899` |
| `air_9.png` | `1f799605e53fc90a1a4a9ce415d0e65fd1b0fe678b1ea3bc0ce5595913d71344` |
| `earth_0.png` | `c3136a3cda09961b615e202db5a8d576c2e2d4a75cffb89db4982447a4860f55` |
| `earth_1.png` | `3556ad112e1f27b4d7d41e412f214bc02a4232c93c2966524d2b7aa3bd2c391e` |
| `earth_2.png` | `305de1a49b45d263db24c80ef17f75fd8e9f0f1cacb27d66e6b05790e074afda` |
| `earth_3.png` | `90900d1442fb66209b4bbd890a4f98d06b0a4be9f3272d851c4b3c47b5634827` |
| `earth_4.png` | `de18a336d690f61aea268f03176ce8ab2331af069bc5f7a27e093e7251a470af` |
| `earth_5.png` | `45af88409998e72e0603eb391689bb0802061159bbe3129a4fc523a06aadb774` |
| `earth_6.png` | `d19d10eb56bf9cf4c2d60d25ca30fed2808f2c6010f524b40658de06a551f76d` |
| `earth_7.png` | `d0f2c8d0bd8a87f80c8872b439680be87231ff447506203b2caf9c928615e51e` |
| `earth_8.png` | `8dbe0abb17847c8487d63961d41948c5ab57ec2350b248791c332edc7719b5e7` |
| `earth_9.png` | `bedbd886865c6aa80ca3698ba40d558e5e3e307a0fd0b192ab27c63733bd9a20` |
| `fire_0.png` | `ebc967ed3a37815b5f2f79868867a9060a07eed2f83069a94a8217baa949ad22` |
| `fire_1.png` | `3f55ea6d8ce7bc3e078054a93a58222d5087fd6a3500c1eac4ac1624628ff06d` |
| `fire_2.png` | `55bc6dc827f727e343a7952396c73a2b03f9b62d0a9549ea6edfd25243015ab2` |
| `fire_3.png` | `140fd4355e3147d0b33c9d3af6269a5602b76381152700a93e9b371028d89d6d` |
| `fire_4.png` | `0f8d4d5e39c9f112a2c817bc4e01b70052e2719336c77f48b965057614595cc2` |
| `fire_5.png` | `75616dad0b2eb6278a8aae3e26a3b4a7f3468a022a3533c3dea9c603320aebea` |
| `fire_6.png` | `9447cb23ee8e14d06f6c7916c6b488ddc564ac9ee95796c41fa1585cdda0b7dd` |
| `fire_7.png` | `322f24a0f985bd6ec85926f56e5517b151e4352dac3b225d30f9091dafdf7e77` |
| `fire_8.png` | `1aa64e0131c08a43d2c764edb469efc6d805fbe2e10804b85e810813f2f50f59` |
| `fire_9.png` | `b029104f380a1764c0d211a226b70f66168d01c962e7edfabcafe78d85475d3d` |
| `inventory_64.png` | `96577f9303dd498a422e5bb863b77da4aa0f00bcf6a9ffdde9ce2b34c10629f9` |
| `inventory_64_press.png` | `0b3f782fa207a58245432cc2bed1771ce0aec738c3990e63659cd388227ca21e` |
| `inventory_64_selected.png` | `662c6972332b7be17909f9119e4f471fb87a8ba6f1f90d8f9c97eca4a7bed064` |
| `items_condensed.png` | `e62069d0b96f1761a80ecc64b392ae2aebe0a8d35f52b19ad0b5ab1619806ef8` |
| `map_shader_100.fs` | `c14b5362754e9d833a8a920d9f3547bf3cee67aeada8e1a0290406aa025ff633` |
| `map_shader_330.fs` | `68adfe17b28dc10c56f5aa2353e9518f99ca63df5ef87a1cecbdb86cc8c841de` |
| `merged_sheet.png` | `9f5cf806090f78960c8a22c98f744ad9aa72ac8767452d805938c5664efd70c2` |
| `neutral_0.png` | `65044cb60515c87f4e79e3a3356dd364f34843b6a7a59da5de48427be0d0e8d6` |
| `neutral_1.png` | `5a921821752b36de06f833d20d901338a7db5b40ae59149762a17435a469cfa6` |
| `neutral_2.png` | `328d9c81e35799268507c303ad7869887830eea18dd3c8ee32acb092fe887d80` |
| `neutral_3.png` | `96ed9524cd823906774fd4979f3c32a5462b412b2d1b9fabb3050b447d489cca` |
| `neutral_4.png` | `04c3f2d74506f1640eafe3ad0486142ceccdf0a18ccad450f51c6b05620372cb` |
| `neutral_5.png` | `00d3516412cfee15f6a64db08b28ec1118a7d8279bf6cc45a6bde03ce995f949` |
| `neutral_6.png` | `2439cdbd7f87e4ffd62fc835eabe584c7f2ccefde9d216b15f169d6fc2b1cbdf` |
| `neutral_7.png` | `3ff55e5e97338b9d28ecd8aff9f06f63769cf2ec458207ac1697ce17a237b864` |
| `neutral_8.png` | `90ba669696ea449d61f6145befd6be6abfee55cbf7afc4320a9e744ab27a65a0` |
| `neutral_9.png` | `e0fdd0fb9ac8a85e6744dcff2a436e93a994ba685c6b61dc188b7413c5d75df2` |
| `nmmo3_weights.bin` | `2bb4a521d2f744e23525baa16a074eb4bc88f1b26b5ad54d7fd571df89887da5` |
| `water_0.png` | `e741b24d786c4a42e812fbbfb80a307e32e4b51ca2f9821ffb2b7b33384812e3` |
| `water_1.png` | `000a2be95a730e24c8540ef7d9689f74d6e85b33f6d15bca4c4184d9546a4b7b` |
| `water_2.png` | `cc050138d1dff14a2e0aa2fc87d4d291bdfabaf3ca9b1fcc893e8859acc3ec2b` |
| `water_3.png` | `7866dd914d3f642a591c37373ae958d58ffe66b5927c0f89fbe23db28784f684` |
| `water_4.png` | `4df069ef99661a2fa1c587c8ddcda39dcd18a477bf86cfd7812b33b1e18ec418` |
| `water_5.png` | `5524fd82eb79c635606c4dbbfcf875a6bcee85440118ee850e8bff8e2e4d66a4` |
| `water_6.png` | `52242987d6dc5778c57603ccf422325c74fd38eb95047d46d424ce5b8f063c1a` |
| `water_7.png` | `42ad2e45935b7e4d24cf06feda8b8db8ebce89fbd5d36a1739a781ea4c90b03f` |
| `water_8.png` | `f3affb26127fb1cc1d2a2e6b3659faed96df23b1b1a1ca38a780588b564c9f2f` |
| `water_9.png` | `32d988f39d4caf4699d2af916ae83de876388a7aa877f6f9d2d6df72dac5408d` |

### Deliberately NOT vendored

- `resources/nmmo3/ManaSeedBody.ttf` — `make_client` calls
  `LoadFont("resources/nmmo3/ManaSeedBody.ttf")` (`nmmo3.h:2508`) but the file
  does not exist at the pinned commit; raylib falls back to its built-in
  default font. Trained-on/shipped-with behavior — do not add a font.
- `resources/nmmo3/nmmo3_help.png` — present upstream but never opened by any
  code at the pin (no `LoadTexture` reference).
- `resources/shared/` — upstream `build.sh --web` preloads it generically for
  every env (`--preload-file resources/shared@resources/shared`), but the
  nmmo3 renderer never opens anything under it (fonts/branding for other
  Ocean envs). Same call as moba: not vendored; the viewer build preloads
  only `resources/nmmo3`.

`vendor/LICENSE-pufferlib` is the upstream repo `LICENSE` file.

## Build toolchain

- emscripten: `emcc 6.0.5-git` (Homebrew). Recorded for build reproducibility;
  the wasm binaries are build outputs (gitignored), not committed.

### Build-time dependency: raylib 5.5 (web, prebuilt)

Not vendored into git; fetched by `sim/build_viewer.sh` into
`build/raylib-web/` (cached) exactly as upstream `build.sh --web` does:

- URL: https://github.com/raysan5/raylib/releases/download/5.5/raylib-5.5_webassembly.zip
- Upstream pins this same artifact (`build.sh`: `RAYLIB_URL=".../5.5"`,
  `RAYLIB_NAME='raylib-5.5_webassembly'`) — a prebuilt emscripten static
  library, no source build.
- zip sha256: recorded/verified by `sim/build_viewer.sh` (`RAYLIB_ZIP_SHA256`).
