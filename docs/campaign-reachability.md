# Campaign reachability and encounter ownership

Campaign encounter locations are discovered from the ROM encounter catalog,
but a catalog entry is not automatically an executable destination. The static
world graph describes physical map connections; campaign reachability overlays
the current campaign facts and registered mandatory checkpoints on that graph.

An encounter destination now has one of these planning states:

- `reachable_now`: the optional encounter may take execution ownership;
- `reachable_after_preparation`: the route exists, but a registered mandatory
  checkpoint must be prepared and cleared first;
- `unreachable`: no route exists in the current ROM graph;
- `unknown`: the current map or required observation is unavailable.

Future encounters remain in the internal policy frontier and retain their
static travel and utility information. Only `reachable_now` encounters may be
selected by the strategic encounter policy or shown in the human-facing
frontier. This lets an encounter remain a valuable future target without
sending the tactical executor toward it before the campaign can actually get
there.

The current trainer registry represents Calvin's first Route 102 battle as a
mandatory checkpoint. Reaching Petalburg is the ROM-backed fact that proves
the player has crossed that checkpoint. Consequently, Route 102 remains a
valid encounter destination while routes beyond it are preparation-gated until
the Petalburg fact is observed. The Emerald access registry also records that
Route 111 requires the first two gym badges. That requirement is evaluated from
the save-backed badge flags, so the route remains a visible future opportunity
while the campaign continues through its required objectives. Unresolved
clearance or badge facts do not open the route.

The same registry records the early story corridor that leads through
Petalburg, Petalburg Woods, Rustboro, and the Rusturf Tunnel, along with later
badge and field-move boundaries. It also records the Surf and Basement Key
requirements for New Mauville and the late-game requirements for Altering
Cave, Sky Pillar, and Victory Road. These are game-specific data entries
consumed by the generic evaluator; adding a new story gate does not require
encounter-policy code to know the map's story.

Map arrival is not a campaign objective when the ROM exposes a meaningful
scene at the destination. The early frontier therefore owns Wally's gym scene
after Petalburg, the Rustboro Devon Goods scene after the Woods, and the
Rusturf Tunnel recovery scene after the goods are stolen. The corresponding
city-visit flags remain observations used to merge route facts and validate
the ROM boundary.

Mandatory battles use the same boundary. Calvin's Route 102 checkpoint is
known from the verified map object, while scripted rivals and story grunts
resolve their concrete party from the active ROM trainer catalog. Readiness
can then mount bounded preparation before the scene capability or promoted
trainer engagement continues.

The reachability assessment is pure and consumes `CampaignState`, ROM-owned
access requirements, and the cached `WorldMapGraph`. It does not read the
emulator or mutate campaign state. The campaign planning signature already
includes the map and campaign facts that can change the result, so future
integration can cache assessments until one of those boundaries changes.

This is the first dynamic gate layer, not complete whole-game gate coverage.
Later work must extend the ROM-owned registry with field-move, story, puzzle,
and later trainer requirements. Those additions belong in verified game-data
definitions and should be evaluated by the same reachability contract rather
than by encounter-specific route rules.

Regression coverage verifies that the current checkpoint map remains open, a
future destination beyond it becomes preparation-gated, clearing the
campaign fact opens the route, and an inaccessible encounter cannot displace a
required progression task.
