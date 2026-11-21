# Baltic landscapes from public data (2026-11-20)

Scripts: `scripts/build_food_layers.py`, `scripts/fetch_ais_huetten.py`,
`scripts/build_ships_from_ais.py`; code in `cenop.landscape.food_layers` and
`cenop.agents.ais_routes`. Provenance of the generated files:
`data/Lithuania/FOOD_PROVENANCE.txt`, `data/Lithuania/SHIPS_PROVENANCE.txt`.

## Lithuania: food

**Published version: interim.** The files in the repository carry the random patches described
here with uniform prey (0.38592 in every month). The SAMBAH-derived prey maps below are kept
outside the repository until their licence is cleared (see "Licence of the SAMBAH data");
`scripts/build_food_layers.py` rebuilds them from the SAMBAH download in minutes. The results
quoted in this section were computed with the SAMBAH maps on the water mask before the sea gaps
were filled (see "Lithuania: sea gaps filled").

**Method (DEPONS 3.2 TRACE 2.6).** Food patches are single 400 m cells chosen at random,
1.6% of the water (8,382 cells since the sea gaps were filled on 2026-11-21; 7,344 before;
seed 0). Monthly prey maps come from seasonal maps of relative
porpoise distribution, interpolated per cell with a + b sin(x) + c cos(x) and scaled to a mean
of 0.38592 over the water. Fitting DEPONS's own Kattegat April, July and October maps this way
reproduces its January to November maps within 0.0005 (its December map differs; correlation
0.98).

**Source.** SAMBAH modelled probability of harbour porpoise detection, May-October and
November-April averages of monthly model maps (Carlen et al. 2018; data Carlen 2017). It is the
only public gridded porpoise distribution for the Baltic Proper; SCANS-IV, van Beest et al.
2025 and Waggitt et al. 2020 do not reach Lithuanian waters.

**Choices made here.**

- The two seasons are placed at the centres of their months (7.5 and 1.5), six months apart,
  so the curve is a cosine peaking in late July.
- SAMBAH covers 54% of the landscape's water. Water deeper than 80 m was outside the SAMBAH
  model and gets the 10th percentile of the values; other gaps (Russian waters, the Curonian
  Lagoon, very shallow water) the nearest value.
- Prey maps are written on the landscape's 1 km grid (SAMBAH's cells are 960 m); patches on
  the 400 m grid, which the loader uses as written.

**Result.** The strong west-east gradient of the data is kept (July means: 0.95 at 17-18 E,
0.009 at 21-22 E; most food around the southern Midsea Bank). With 2000 animals the mean
energy is 10.0 at day 10 and 9.1 at day 120, 1,913 alive at day 120; with the placeholder
food it stayed at the 19.9 cap.

## Lithuania: ships

**Source.** Raw AIS for the whole Baltic from aisstream.io, 29 July to 27 October 2024 (Huetten
2025, figshare, CC BY 4.0). Danish Maritime Authority AIS reaches these waters only on some
days, and EMODnet, EMSA and HELCOM publish densities, not tracks.

**Method.** 15 days, 12-27 August 2024, positions inside the landscape and on its water (67%
of them after the sea gaps were filled, see below; Klaipeda harbour and its channel, narrower
than a 1 km cell, are left out, where a ship would emit nothing). Every passage of every ship becomes a route sampled at
30-minute ticks, placed in time by its first-buoy pause and padded to a 721-tick cycle, the
encoding of DEPONS's Kattegat file. Passages are split at gaps over 2 hours; moored ships and
passages under one hour are dropped.

**Result.** 3,469 passages of 1,201 vessels. AIS type codes
become JOMOPANS classes: cargo (70-79) becomes Bulker, since AIS does not separate bulk,
container and vehicle carriers; passenger ships include ferries and cruise ships. Lengths are
known for 1,621 passages from Danish AIS static records; the rest get the median length of
their class in DEPONS's Kattegat file. In a 10-day run with 2000 animals 4.3% of animal-ticks
are deterred by ships (Kattegat: 19%, with denser traffic through narrower straits).

**Limits.** One summer fortnight (the source has no winter); aisstream.io's own terms were not
checked; the class mapping cannot tell cargo ship types apart.

## Lithuania: sea gaps filled (2026-11-21)

12.4% of the sea in the Lithuania box (10,409 km2) had no depth, so the model treated it as
land: a strip along the whole northern edge (57.0-57.5 N, about 5,400 km2), parts of the Gulf
of Gdansk and of the Curonian Lagoon, and 240 scattered holes. 873 water cells bordered this
missing sea, an artificial shore across open water, and ship noise stopped at it. Where the
landscape had depths they agree with EMODnet's (median difference 0.23 m, 91% of cells within
1 m, largest 11 m), so the gaps were filled from the
same EMODnet DTM (`scripts/build_baltic_bathymetry.py --fill-gaps-of`), keeping every earlier
water cell. The Curonian Lagoon stays a separate water body (1,754 km2 now, its Russian part
included); the main sea body grows from 71,979 to 79,143 km2. Distance to coast, blocks (now 40 km
squares; previously one block), salinity, sediment, food patches and ships were rebuilt on the
new water mask.

## Movement salinity (2026-11-21)

The DEPONS movement model draws step lengths and turning angles that depend on salinity
(`log10(step) = 0.35 prev + 0.0005 depth - 0.02 salinity + N(1.25, 0.15)`), fitted to
porpoises tagged in the eastern North Sea, where the map salinity is 31-35. DEPONS moves with
a fixed 34.07 on Kattegat because the model is not valid at much lower salinity (TRACE 3.2,
section 3.2.2). With the Baltic map salinity (about 7) the draws sat near the maximum: mean
steps of 10 cells (4 km) per 30 minutes against 2 on Kattegat and the North Sea, and about 60%
of draws near the cap. Movement on Lithuania and CentralBaltic now uses the fixed value too
(`movement_salinity="auto"`; `"map"` restores the map). Sound propagation still uses the map.

| 2000 animals, ships, 60 days | Mean step (cells) | Deterred animal-ticks | Mean energy, day 60 |
|---|---|---|---|
| Lithuania, map salinity | 10.1 | 3.5% | 9.89 |
| Lithuania, fixed (default) | 2.3 | 3.3% | 9.30 |
| CentralBaltic, map salinity | 10.1 | 5.5% | 11.41 |
| CentralBaltic, fixed (default) | 2.3 | 5.1% | 10.82 |

Results quoted elsewhere in this note were computed with the map salinity.

## Licence of the SAMBAH data: not cleared

The SAMBAH data record (Swedish National Data Service) names no licence; HELCOM's copy of the
same maps allows non-commercial use with citation. The CENOP repository is public, so the
derived prey maps are not published before the data owner confirms (contact on the record:
Ida Carlen, Coalition Clean Baltic). The same contact could supply the twelve monthly SAMBAH
maps, which would replace the two-season interpolation.

## Salinity and sediment (both landscapes)

Lithuania's salinity was random around 7.5 and its sediment a single value; CentralBaltic's
were generator output. Both now come from open data (`scripts/build_baltic_environment.py`):

- **Salinity:** Copernicus Marine BALTICSEA_MULTIYEAR_PHY_003_011, monthly means for 2021 at the
  top model level (0.5 m), the product and year DEPONS used for Kattegat
  (`scripts/fetch_copernicus_salinity.py`; needs a Copernicus Marine account, set up with
  `copernicusmarine login`). Lithuania 0.01-7.9, CentralBaltic 0.01-15.4 (near-fresh water in
  lagoons and river mouths; higher towards the Arkona Sea). On the overlap with DEPONS's
  Kattegat file near the Sound the two agree: January medians 8.53 and 8.53, July 7.31 and
  7.59, correlation 0.97-0.99. Open alternative without an account: the Baltic & North Sea
  Climatology (BNSC v1.1, 2006-2015, 0.25 degree, CC BY 4.0), which `build_baltic_environment.py`
  also reads; it agrees with NOAA's World Ocean Atlas 2023 offshore and gives slightly higher
  summer values (July means 6.9-7.1 against 6.5-6.7).
- **Sediment:** median grain size (phi) from EMODnet Geology seabed substrate (1:1M, 1:250k,
  1:100k and the Lithuanian 1:50k multiscale map; finer maps overwrite coarser), CC BY 4.0,
  with the per-class values DEPONS's Kattegat file uses (TRACE Appendix 1, Table S1): Mud
  7.142, sandy Mud 4.40, muddy Sand 3.155, Sand 2.277, Coarse 1.218, Mixed 1.495, Rock -8.
  About 30% of the water is mapped only as "Mud to muddy Sand", for which the table has no
  value; it gets the sandy-Mud value 4.40, the middle of the three classes it spans (a choice
  made here). The maps cover 97-99.9% of the water; lagoons take the nearest value. Swedish
  polygons marked "restricted" at 1:250k and 1:100k are covered by the 1:1M map only.

## CentralBaltic: rebuilt

The old CentralBaltic (a placeholder since the first commit) was not where its header said:
its download script stored EMODnet depths fetched in latitude and longitude (53.9-59.5 N,
13-22 E) under an EPSG:3035 header without reprojecting, so over a million "water" cells lay
over inland Poland and its deepest cells sat at about 52 N.

The new landscape covers the area the old data came from: 615 x 677 cells of 1 km from
(4491000, 3425000) in EPSG:3035, about 53.4-60 N and 12.6-23.9 E, the Baltic Proper from
Bornholm to north of Gotland.

- **Depth:** EMODnet DTM, reprojected with average resampling (`scripts/build_baltic_bathymetry.py`).
  201,414 km2 of water, deepest 447 m, median 56 m. Checked: Bornholm, Gotland and Oland are
  land; the Gotland Deep is 246 m and the Bornholm Basin 92 m. On the overlap with Lithuania every
  Lithuania water cell is water here and depths agree (median difference 0.2 m).
- **Distance to coast** in metres to the nearest non-water cell; **blocks** of 40 km (168).
- **Salinity and sediment** as above; **food**: random patches with uniform prey in the published
  files; SAMBAH-derived prey (59% of the water covered) built but held back, as for Lithuania; patches
  on the 400 m grid; **ships** from the same 15 days of 2024 AIS (10,245 passages of 2,139
  vessels; 79% of the AIS positions fall on the landscape's water).
- In a three-day run with 2000 animals ships deter 8.1% of animal-ticks; a tick takes about
  17 ms with ships.
- Size: 117 MB, against 842 MB for the old 250 m files.

The scripts that built the old grid (`generate_central_baltic.py`, `download_emodnet_*.py`,
`fill_centralbaltic_data_gap.py`) are removed.

## References

- Carlen, I., et al. (2018). Basin-scale distribution of harbour porpoises in the Baltic Sea
  provides basis for effective conservation actions. *Biological Conservation* 226, 42-53.
  https://doi.org/10.1016/j.biocon.2018.06.031
- Carlen, I. (2017). SAMBAH: Harbour porpoise distribution in the Baltic Sea (v1.0). Swedish
  National Data Service. https://doi.org/10.5879/fmyy-wf63
- Huetten, M. (2025). AIS data of the Baltic Sea from aisstream.io, 29 July - 27 October 2024.
  figshare. https://doi.org/10.6084/m9.figshare.29062715
- Danish Maritime Authority, AIS data (http://aisdata.ais.dk): contains data from the Danish
  Maritime Authority.
- EMODnet Bathymetry Consortium: EMODnet Digital Bathymetry (DTM). https://emodnet.ec.europa.eu/en/bathymetry
- EMODnet Geology: seabed substrate (Folk classification). https://emodnet.ec.europa.eu/en/geology
- Copernicus Marine Service: Baltic Sea Physics Reanalysis, BALTICSEA_MULTIYEAR_PHY_003_011.
  https://doi.org/10.48670/moi-00013
- Baltic & North Sea Climatology (BNSC) v1.1, University of Hamburg, ICDC.
  https://icdc.cen.uni-hamburg.de/thredds/catalog/ftpthredds/bnsc/ocean/v1.1/catalog.html
