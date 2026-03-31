# Viewer Retarget Notes

Stable default setup:

- Default model: `MoonGirl.vrm`
- Default retarget path: `skeletonutils-skinnedmesh + live-delta`
- Console mode: `minimal`
- VRM direct retarget: disabled by default
- Rig profile cache key: `vid2model.rigProfiles.v16`

Useful runtime flags:

```js
window.__vid2modelForceLiveDelta = null
window.__vid2modelUseVrmDirect = false
window.__vid2modelSetDiagMode("minimal")
```

Current practical recommendation:

- Use `MoonGirl.vrm` for normal testing and demos.
- Enable `window.__vid2modelUseVrmDirect = true` only for VRM retarget experiments.

Current status:

- Generic stable path works best with `MoonGirl.vrm`.
- `retarget-vrm.js` is experimental and kept for future VRM-specific axis/sign calibration.
- Alicia-specific facing fixes are no longer part of the default path.
