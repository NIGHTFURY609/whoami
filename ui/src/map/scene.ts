import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import type { Pose } from '../source/api'
import type { CloudFrame, GridFrame, TrajectoryFrame } from './codec'
import { buildGridTexture, ribbonStrip } from './geometry'
import type { LiveMode } from './mapToggles'
import {
  CAR_LENGTH_M, CAR_WIDTH_M, cloudBounds, gridQuad, groundGridFor, heatHeightBand, homeView, pointBounds, unionBounds,
  yawOf, type Bounds, type PlanarPose, type View,
} from './scene-math'
import { EMPTY_Z, Terrain } from './terrain'

// The 3D map (map view), as in the owner's reference picture: the live depth scan as a height heat map (blue low to
// red high), either merged into a rolling terrain around the robot or as the current scan only; the cost grid with
// its free cells painted as the pathway; the travelled path as a ribbon the car's width; and the car itself, over a
// ground grid. A plain class in the manner of glyph-ring's RingScene: made once per mount on its container, fed
// through setters, released by dispose(). No React in here. World frame = the map frame: metres, x forward/east,
// y left/north, z up.
//
// Rendering is on demand only: one requestAnimationFrame is scheduled when something changed (a camera move, a
// setter, a resize) and nothing runs in between - this GPU is shared with the segmentation and depth networks.
//
// Live points, and what they cost. Heights are coloured in the vertex shader from z (no colour buffer, no per-frame
// colour pass on the main thread). Each mode keeps one GPU buffer that is updated in place, never re-allocated per
// frame, and switching mode releases the other mode's buffer:
//   terrain  one float per grid cell (terrain.ts): the shader derives the cell's x and y from gl_VertexID and the
//            window origin, and hides empty cells. A scan uploads only the range of cells it wrote.
//   scan     the latest scan's x y z, in a buffer that grows (doubling) when a scan is bigger than any before it.
//
// Colour space. The heat ramp and every hex colour are sRGB values written straight to the sRGB canvas: the point
// shader includes neither three's colour-space conversion nor tone mapping, and built-in materials take hex colours
// as sRGB and give them back unchanged. The cost-grid texture is tagged SRGBColorSpace, so the sampler decodes it
// and the output encodes it again: its bytes too are shown as they are.
//
// Draw order. Objects in the opaque pass are drawn in renderOrder order:
//   -1 ground grid lines, no depth write (whatever is drawn later covers them)
//    1 live points, depth tested
//    2 cost grid, draped flat: no depth test, blended (custom blending keeps it in the opaque pass) at GRID_OPACITY,
//      so the painted pathway and the halo tint the terrain instead of hiding under it
//    3 travelled ribbon, no depth test, blended
//    4 the car, no depth test: never hidden inside the terrain

export interface LayerVisibility { live: boolean; trajectory: boolean; grid: boolean }
type SceneLayer = keyof LayerVisibility
const SCENE_LAYERS: readonly SceneLayer[] = ['live', 'trajectory', 'grid']

const ORDER = { ground: -1, live: 1, grid: 2, ribbon: 3, car: 4 } as const

const BACKGROUND = '#03100c' // --bg
const GROUND_LINE = '#1f4637' // between --line and --line-strong
const PATH_COLOR = '#ff9a3c' // the travelled ribbon and the car, the reference picture's orange
const GRID_LIFT_M = 0.02 // the cost grid sits just above z = 0
const GRID_OPACITY = 0.5
const RIBBON_OPACITY = 0.55
const RIBBON_LIFT_M = 0.01
const CAR_LIFT_M = 0.03
const POINT_MIN_PX = 1.5 // CSS pixels; scaled by the device pixel ratio
const POINT_MAX_PX = 14
const TERRAIN_SPLAT = 1.25 // terrain points against their cell: a little overlap closes the gaps between cells
const SCAN_SPLAT = 1.6 // scan points against the gateway's thinning voxel

// Turbo-like ramp, sRGB out (Google's polynomial fit of Turbo): dark blue, blue, cyan, green, yellow, orange, red.
const HEAT_GLSL = /* glsl */ `
  uniform vec2 uBand; // z at the low and the high end of the ramp
  vec3 heat(float z) {
    float x = clamp((z - uBand.x) / max(uBand.y - uBand.x, 1e-3), 0.0, 1.0);
    const vec4 kr4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const vec4 kg4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const vec4 kb4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const vec2 kr2 = vec2(-152.94239396, 59.28637943);
    const vec2 kg2 = vec2(4.27729857, 2.82956604);
    const vec2 kb2 = vec2(-89.90310912, 27.34824973);
    vec4 v4 = vec4(1.0, x, x * x, x * x * x);
    vec2 v2 = v4.zw * v4.z;
    return clamp(vec3(dot(v4, kr4) + dot(v2, kr2), dot(v4, kg4) + dot(v2, kg2), dot(v4, kb4) + dot(v2, kb2)), 0.0, 1.0);
  }
`

const SIZE_GLSL = /* glsl */ `
  uniform float uSizeM;   // point size in metres
  uniform float uPxPerM;  // drawing-buffer pixels per metre at unit view depth
  uniform vec2 uPxRange;  // clamp, in drawing-buffer pixels
  float pointSize(vec4 mv) {
    return clamp(uSizeM * uPxPerM / max(-mv.z, 0.001), uPxRange.x, uPxRange.y);
  }
`

const SCAN_VERTEX = /* glsl */ `
  ${HEAT_GLSL}
  ${SIZE_GLSL}
  varying vec3 vColor;
  void main() {
    vColor = heat(position.z);
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = pointSize(mv);
  }
`

// One float per cell (aZ); the cell's centre comes from its index in the row-major window.
const TERRAIN_VERTEX = /* glsl */ `
  ${HEAT_GLSL}
  ${SIZE_GLSL}
  attribute float aZ;
  uniform vec2 uOrigin;  // map x, y of the outer corner of cell 0
  uniform float uCell;   // metres
  uniform int uSize;     // cells per side
  varying vec3 vColor;
  void main() {
    if (aZ < ${(EMPTY_Z / 2).toFixed(1)}) { // an empty cell: outside the clip volume, no size
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
      gl_PointSize = 0.0;
      vColor = vec3(0.0);
      return;
    }
    vec2 cell = vec2(float(gl_VertexID % uSize), float(gl_VertexID / uSize)) + 0.5;
    vec3 p = vec3(uOrigin + cell * uCell, aZ);
    vColor = heat(aZ);
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = pointSize(mv);
  }
`

// sRGB out as computed: no colour-space conversion, no tone mapping (see the header).
const COLOR_FRAGMENT = /* glsl */ `
  varying vec3 vColor;
  void main() {
    gl_FragColor = vec4(vColor, 1.0);
  }
`

const pointUniforms = () => ({
  uSizeM: { value: 0.05 },
  uPxPerM: { value: 1 },
  uPxRange: { value: new THREE.Vector2(POINT_MIN_PX, POINT_MAX_PX) },
  uBand: { value: new THREE.Vector2(...heatHeightBand(null)) },
})

const pointSizeM = (spacingM: number) => (Number.isFinite(spacingM) && spacingM > 0 ? Math.min(0.5, Math.max(0.01, spacingM)) : 0.03)

// Straight-alpha blending that keeps a material in the opaque pass, so renderOrder alone places it.
const overlayBlending = {
  transparent: false,
  blending: THREE.CustomBlending,
  blendEquation: THREE.AddEquation,
  blendSrc: THREE.SrcAlphaFactor,
  blendDst: THREE.OneMinusSrcAlphaFactor,
  depthTest: false,
  depthWrite: false,
  toneMapped: false,
} as const

export interface SceneInfo { terrainCells: number | null } // null outside the terrain mode

export class MapScene {
  // Called after the terrain changed (not more than once per live frame).
  onInfo: ((info: SceneInfo) => void) | null = null

  private container: HTMLElement
  private renderer: THREE.WebGLRenderer
  private controls: OrbitControls
  private scene = new THREE.Scene()
  private camera = new THREE.PerspectiveCamera(50, 1, 0.05, 4000)

  private scanMat = new THREE.ShaderMaterial({ vertexShader: SCAN_VERTEX, fragmentShader: COLOR_FRAGMENT, uniforms: pointUniforms() })
  private terrainMat = new THREE.ShaderMaterial({
    vertexShader: TERRAIN_VERTEX,
    fragmentShader: COLOR_FRAGMENT,
    uniforms: { ...pointUniforms(), uOrigin: { value: new THREE.Vector2() }, uCell: { value: 0.05 }, uSize: { value: 1 } },
  })
  private ribbonMat = new THREE.MeshBasicMaterial({ color: PATH_COLOR, opacity: RIBBON_OPACITY, side: THREE.DoubleSide, ...overlayBlending })
  private gridMat = new THREE.MeshBasicMaterial({ map: null, side: THREE.DoubleSide, opacity: GRID_OPACITY, ...overlayBlending })
  private carFillMat = new THREE.MeshBasicMaterial({ color: PATH_COLOR, opacity: 0.35, side: THREE.DoubleSide, ...overlayBlending })
  private carLineMat = new THREE.LineBasicMaterial({ color: PATH_COLOR, depthTest: false, depthWrite: false, toneMapped: false })

  private live = new THREE.Points(new THREE.BufferGeometry(), this.scanMat)
  private ribbon = new THREE.Mesh(new THREE.BufferGeometry(), this.ribbonMat)
  private grid = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), this.gridMat) // unit plane, scaled to the grid
  private gridTexture: THREE.DataTexture | null = null
  private car = new THREE.Group()
  private ground: THREE.GridHelper | null = null
  private groundKey = ''

  private liveMode: LiveMode = 'terrain'
  private terrain: Terrain | null = null
  private zAttr: THREE.BufferAttribute | null = null // terrain heights, on the GPU
  private scanAttr: THREE.BufferAttribute | null = null // scan positions, grown on demand

  // What is shown, per layer: the frame it was built from (so an unchanged frame is not rebuilt), whether it has
  // anything to draw, and its bounds (ground grid extent and camera framing).
  private frames: { live: CloudFrame | null; trajectory: TrajectoryFrame | null; grid: GridFrame | null } =
    { live: null, trajectory: null, grid: null }
  private visible: LayerVisibility = { live: true, trajectory: true, grid: true }
  private drawable: Record<SceneLayer, boolean> = { live: false, trajectory: false, grid: false }
  private bounds: Record<SceneLayer, Bounds | null> = { live: null, trajectory: null, grid: null }
  private robot: PlanarPose | null = null
  private robotKey = ''

  private framed = false // the camera has been placed on data (or the operator moved it first); never again on data
  private frameId = 0
  private disposed = false

  // Throws when WebGL cannot start; the caller shows a message instead.
  constructor(container: HTMLElement) {
    this.container = container
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'low-power' })
    const el = this.renderer.domElement
    try {
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
      this.renderer.setClearColor(BACKGROUND, 1)
      el.style.position = 'absolute'
      el.style.inset = '0'
      el.style.width = '100%'
      el.style.height = '100%'
      el.style.display = 'block'
      el.style.touchAction = 'none'
      container.appendChild(el)

      this.camera.up.set(0, 0, 1) // z up, set before the controls read it
      this.controls = new OrbitControls(this.camera, el)
      this.controls.screenSpacePanning = false // pan along the ground
      this.controls.maxPolarAngle = Math.PI * 0.495 // stay above the ground
      this.controls.minDistance = 0.3
      this.controls.maxDistance = 2000
      this.controls.addEventListener('change', this.invalidate)
      this.controls.addEventListener('start', this.onOperatorMove)
      el.addEventListener('webglcontextrestored', this.invalidate)
    } catch (e) {
      this.renderer.dispose()
      this.renderer.forceContextLoss()
      el.remove()
      throw e
    }

    this.live.renderOrder = ORDER.live
    this.live.frustumCulled = false // its geometry has no position bounds in the terrain mode
    this.grid.renderOrder = ORDER.grid
    this.ribbon.renderOrder = ORDER.ribbon
    this.ribbon.frustumCulled = false
    this.buildCar()
    this.car.visible = false
    for (const o of [this.live, this.ribbon, this.grid]) o.visible = false
    this.scene.add(this.live, this.grid, this.ribbon, this.car)

    this.applyView(homeView(null, null))
    this.updateGround()
    this.invalidate()
  }

  // ---- data -------------------------------------------------------------------------------------
  // The live depth scan: merged into the terrain, or shown as it is, by the live mode.
  setLive(frame: CloudFrame | null) {
    if (this.disposed || frame === this.frames.live) return
    this.frames.live = frame
    this.showLive(frame)
  }

  // terrain | scan. The mode left behind gives its buffer back; the terrain starts from the scan held.
  setLiveMode(mode: LiveMode) {
    if (this.disposed || mode === this.liveMode) return
    this.liveMode = mode
    this.releaseLiveBuffers()
    this.showLive(this.frames.live)
  }

  // The travelled path as a ribbon the car's width, from the trajectory to the car's current position.
  setTrajectory(frame: TrajectoryFrame | null) {
    if (this.disposed || frame === this.frames.trajectory) return
    this.frames.trajectory = frame
    this.rebuildRibbon()
  }

  // The cost grid as a texture on a quad just above the ground, placed by its origin, resolution and yaw, sampled
  // nearest-neighbour so every cell stays a crisp square. Free cells are the painted pathway (buildGridTexture).
  setGrid(frame: GridFrame | null) {
    if (this.disposed || frame === this.frames.grid) return
    this.frames.grid = frame
    this.gridTexture?.dispose()
    this.gridTexture = null
    const q = frame ? gridQuad(frame) : null
    const max = this.renderer.capabilities.maxTextureSize
    if (frame && q && frame.width <= max && frame.height <= max) {
      const rgba = buildGridTexture(frame)
      // Row 0 of the texture is grid row 0 (at the origin); with flipY off it lands at v = 0, the quad's origin side.
      const tex = new THREE.DataTexture(
        new Uint8Array(rgba.buffer, rgba.byteOffset, rgba.byteLength), frame.width, frame.height, THREE.RGBAFormat, THREE.UnsignedByteType,
      )
      tex.colorSpace = THREE.SRGBColorSpace
      tex.magFilter = THREE.NearestFilter
      tex.minFilter = THREE.NearestFilter
      tex.generateMipmaps = false
      tex.flipY = false
      tex.needsUpdate = true
      this.gridTexture = tex
      this.grid.position.set(q.cx, q.cy, GRID_LIFT_M)
      this.grid.rotation.set(0, 0, q.yaw)
      this.grid.scale.set(q.sizeX, q.sizeY, 1)
    }
    if ((this.gridMat.map === null) !== (this.gridTexture === null)) this.gridMat.needsUpdate = true // with/without a map is another program
    this.gridMat.map = this.gridTexture
    this.layerChanged('grid', this.gridTexture !== null, this.gridTexture && q ? q.bounds : null)
  }

  // The car (map -> base_link). Hidden while the pose is unavailable.
  setPose(pose: Pose | null) {
    if (this.disposed) return
    const raw = pose?.available ? [pose.x, pose.y, pose.z, pose.qx, pose.qy, pose.qz, pose.qw] : []
    const v = raw.length === 7 && raw.every((n): n is number => n !== null && Number.isFinite(n)) ? raw : null
    const key = v ? v.join(',') : ''
    if (key === this.robotKey) return // unchanged: a robot standing still costs no frames
    this.robotKey = key
    if (v) {
      const [x, y, z, qx, qy, qz, qw] = v
      this.car.position.set(x, y, z + CAR_LIFT_M)
      this.car.quaternion.set(qx, qy, qz, qw).normalize()
      this.robot = { x, y, z, yaw: yawOf(qx, qy, qz, qw) }
    } else {
      this.robot = null
    }
    this.car.visible = v !== null
    const band = heatHeightBand(this.robot?.z ?? null)
    for (const m of [this.scanMat, this.terrainMat]) (m.uniforms.uBand.value as THREE.Vector2).set(band[0], band[1])
    if (this.robot && this.terrain?.follow(this.robot.x, this.robot.y)) this.uploadTerrain(null)
    this.rebuildRibbon() // the ribbon runs up to the car
    this.updateGround()
    this.invalidate()
  }

  setLayers(visibility: LayerVisibility) {
    if (this.disposed) return
    this.visible = { ...visibility }
    this.applyVisibility()
    this.invalidate()
  }

  // Back to the home view: behind and above the robot (or over the data), as on the first data.
  resetView() {
    if (this.disposed) return
    this.framed = true
    this.applyView(homeView(this.robot, this.dataBounds()))
  }

  resize(width: number, height: number) {
    if (this.disposed || width <= 0 || height <= 0) return
    this.renderer.setSize(width, height, false)
    this.camera.aspect = width / height
    this.camera.updateProjectionMatrix()
    const dpr = this.renderer.getPixelRatio()
    const pxPerM = (height * dpr) / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2))
    for (const m of [this.scanMat, this.terrainMat]) {
      m.uniforms.uPxPerM.value = pxPerM
      ;(m.uniforms.uPxRange.value as THREE.Vector2).set(POINT_MIN_PX * dpr, POINT_MAX_PX * dpr)
    }
    this.invalidate()
  }

  dispose() {
    if (this.disposed) return
    this.disposed = true
    this.onInfo = null
    cancelAnimationFrame(this.frameId)
    this.frameId = 0
    const el = this.renderer.domElement
    this.controls.removeEventListener('change', this.invalidate)
    this.controls.removeEventListener('start', this.onOperatorMove)
    this.controls.dispose()
    el.removeEventListener('webglcontextrestored', this.invalidate)
    for (const o of [this.live, this.ribbon, this.grid]) o.geometry.dispose()
    this.car.traverse((o) => { if (o instanceof THREE.Mesh || o instanceof THREE.Line) o.geometry.dispose() })
    this.ground?.dispose()
    for (const m of [this.scanMat, this.terrainMat, this.ribbonMat, this.gridMat, this.carFillMat, this.carLineMat]) m.dispose()
    this.gridTexture?.dispose()
    this.scene.clear()
    this.renderer.dispose()
    this.renderer.forceContextLoss() // release the context now, not whenever the canvas is collected
    if (el.parentNode === this.container) this.container.removeChild(el)
    // drop the frames and the terrain, which hold the fetched and merged buffers
    this.frames = { live: null, trajectory: null, grid: null }
    this.terrain = null
    this.zAttr = null
    this.scanAttr = null
  }

  // ---- internals --------------------------------------------------------------------------------
  private invalidate = () => {
    if (this.disposed || this.frameId !== 0) return
    this.frameId = requestAnimationFrame(this.draw)
  }

  private draw = () => {
    this.frameId = 0
    if (this.disposed) return
    this.renderer.render(this.scene, this.camera)
  }

  // The operator took the camera before any data arrived: do not move it when the data comes.
  private onOperatorMove = () => {
    this.framed = true
  }

  private showLive(frame: CloudFrame | null) {
    const has = frame !== null && frame.count > 0
    if (this.liveMode === 'terrain') {
      if (has) this.mergeIntoTerrain(frame)
      this.layerChanged('live', this.terrain !== null && this.terrain.filled > 0, has ? cloudBounds(frame) : this.bounds.live)
      this.onInfo?.({ terrainCells: this.terrain?.filled ?? 0 })
    } else {
      if (has) this.showScan(frame)
      this.live.geometry.setDrawRange(0, has ? frame.count : 0)
      this.layerChanged('live', has, has ? cloudBounds(frame) : null)
      this.onInfo?.({ terrainCells: null })
    }
  }

  private mergeIntoTerrain(frame: CloudFrame) {
    if (!this.terrain) {
      const t = new Terrain()
      const g = new THREE.BufferGeometry()
      this.zAttr = new THREE.BufferAttribute(t.z, 1)
      this.zAttr.setUsage(THREE.DynamicDrawUsage)
      g.setAttribute('aZ', this.zAttr)
      g.setDrawRange(0, t.size * t.size) // no position attribute: the range says how many points there are
      this.replaceGeometry(g)
      this.live.material = this.terrainMat
      this.terrainMat.uniforms.uCell.value = t.cellM
      this.terrainMat.uniforms.uSize.value = t.size
      this.terrainMat.uniforms.uSizeM.value = t.cellM * TERRAIN_SPLAT
      this.terrain = t
      if (this.robot) t.follow(this.robot.x, this.robot.y)
    }
    const t = this.terrain
    const moved = this.robot ? t.follow(this.robot.x, this.robot.y) : false
    const dirty = t.ingest(frame.xyz, frame.count)
    this.uploadTerrain(moved ? null : dirty, moved || dirty !== null)
  }

  // Uploads the terrain heights: the given range, or (null) the whole buffer and the origin with it.
  private uploadTerrain(range: { start: number; end: number } | null, changed = true) {
    const t = this.terrain
    const attr = this.zAttr
    if (!t || !attr) return
    ;(this.terrainMat.uniforms.uOrigin.value as THREE.Vector2).set(t.originX, t.originY)
    if (!changed) return
    attr.clearUpdateRanges()
    if (range) attr.addUpdateRange(range.start, range.end - range.start)
    attr.needsUpdate = true // no range = the whole buffer
    this.invalidate()
  }

  private showScan(frame: CloudFrame) {
    const need = 3 * frame.count
    let attr = this.scanAttr
    if (!attr || attr.array.length < need) {
      // grow by doubling, so a stream of slightly bigger scans re-allocates rarely
      const cap = Math.max(need, attr ? 2 * attr.array.length : 0)
      attr = new THREE.BufferAttribute(new Float32Array(cap), 3)
      attr.setUsage(THREE.DynamicDrawUsage)
      const g = new THREE.BufferGeometry()
      g.setAttribute('position', attr)
      this.replaceGeometry(g)
      this.scanAttr = attr
      this.live.material = this.scanMat
    }
    ;(attr.array as Float32Array).set(frame.xyz.subarray(0, need))
    attr.clearUpdateRanges()
    attr.addUpdateRange(0, need)
    attr.needsUpdate = true
    this.scanMat.uniforms.uSizeM.value = pointSizeM(frame.spacingM) * SCAN_SPLAT
  }

  // Gives both live buffers back (the CPU copies with the terrain, the GPU ones with the geometry).
  private releaseLiveBuffers() {
    this.terrain = null
    this.zAttr = null
    this.scanAttr = null
    this.replaceGeometry(new THREE.BufferGeometry())
    this.layerChanged('live', false, null)
  }

  private replaceGeometry(g: THREE.BufferGeometry) {
    this.live.geometry.dispose() // its GPU buffers are released as the new one takes its place
    this.live.geometry = g
  }

  // The trajectory's positions plus the car's position, as a ribbon CAR_WIDTH_M wide.
  private rebuildRibbon() {
    const f = this.frames.trajectory
    const n = f ? f.count : 0
    const tail = this.robot ? 1 : 0
    const path = new Float32Array(3 * (n + tail))
    for (let i = 0; i < n; i++) path.set(f!.poses.subarray(7 * i, 7 * i + 3), 3 * i)
    if (this.robot) path.set([this.robot.x, this.robot.y, this.robot.z], 3 * n)
    const r = ribbonStrip(path, CAR_WIDTH_M, RIBBON_LIFT_M)
    const g = new THREE.BufferGeometry()
    let bounds: Bounds | null = null
    if (r) {
      g.setAttribute('position', new THREE.BufferAttribute(r.positions, 3))
      g.setIndex(new THREE.BufferAttribute(r.index, 1))
      g.computeBoundingBox()
      const b = g.boundingBox!
      bounds = unionBounds([pointBounds(b.min.x, b.min.y, b.min.z), pointBounds(b.max.x, b.max.y, b.max.z)])
    }
    this.ribbon.geometry.dispose()
    this.ribbon.geometry = g
    this.layerChanged('trajectory', r !== null, bounds)
  }

  // The car, in its own frame (x forward): a translucent body, its outline and a heading tick to the front.
  private buildCar() {
    const L = CAR_LENGTH_M
    const W = CAR_WIDTH_M
    const body = new THREE.Mesh(new THREE.PlaneGeometry(L, W), this.carFillMat)
    const outline = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(L / 2, W / 2, 0), new THREE.Vector3(-L / 2, W / 2, 0),
      new THREE.Vector3(-L / 2, -W / 2, 0), new THREE.Vector3(L / 2, -W / 2, 0),
    ]), this.carLineMat)
    const heading = new THREE.Line(new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0), new THREE.Vector3(L / 2 + 0.12, 0, 0),
    ]), this.carLineMat)
    for (const o of [body, outline, heading]) {
      o.renderOrder = ORDER.car
      o.frustumCulled = false
      this.car.add(o)
    }
  }

  private layerChanged(layer: SceneLayer, drawable: boolean, bounds: Bounds | null) {
    this.drawable[layer] = drawable
    this.bounds[layer] = drawable ? bounds : null
    this.applyVisibility()
    if (layer !== 'live') this.updateGround() // the live scan moves every frame; the grid follows the map instead
    if (!this.framed && drawable) {
      this.framed = true
      this.applyView(homeView(this.robot, this.dataBounds()))
    }
    this.invalidate()
  }

  private applyVisibility() {
    const objects: Record<SceneLayer, THREE.Object3D> = { live: this.live, trajectory: this.ribbon, grid: this.grid }
    for (const l of SCENE_LAYERS) objects[l].visible = this.visible[l] && this.drawable[l]
  }

  private robotBounds = () => (this.robot ? pointBounds(this.robot.x, this.robot.y, this.robot.z) : null)

  private dataBounds(): Bounds | null {
    return unionBounds([...SCENE_LAYERS.map((l) => this.bounds[l]), this.robotBounds()])
  }

  // Re-sizes / re-centres the ground grid to the map (not the live scan) and the robot; rebuilt only when it changes.
  private updateGround() {
    const g = groundGridFor(unionBounds([this.bounds.trajectory, this.bounds.grid, this.robotBounds()]))
    const key = `${g.cx}:${g.cy}:${g.size}:${g.cell}`
    if (key === this.groundKey) return
    this.groundKey = key
    if (this.ground) {
      this.scene.remove(this.ground)
      this.ground.dispose()
    }
    const grid = new THREE.GridHelper(g.size, Math.round(g.size / g.cell), GROUND_LINE, GROUND_LINE)
    grid.rotation.x = Math.PI / 2 // GridHelper lies in x-z; the map's ground is x-y
    grid.position.set(g.cx, g.cy, 0)
    grid.renderOrder = ORDER.ground
    ;(grid.material as THREE.LineBasicMaterial).depthWrite = false
    this.scene.add(grid)
    this.ground = grid
    this.invalidate()
  }

  private applyView(v: View) {
    this.camera.position.set(v.position.x, v.position.y, v.position.z)
    this.controls.target.set(v.target.x, v.target.y, v.target.z)
    this.controls.update() // re-reads the camera against the target and emits 'change', which schedules a frame
    this.invalidate()
  }
}
