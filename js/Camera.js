// ----------------------------------------------------------------------------

import { mat4, vec4, vec3, quat } from 'https://cdn.skypack.dev/gl-matrix';

// ----------------------------------------------------------------------------

export class Camera {
  static kDefaultFOV = (90 * Math.PI) / 180.0;
  static kDefaultSize = 512;
  static kDefaultNear = 0.1;
  static kDefaultFar = 500.0;

  constructor(controller = null) {
    this.controller_ = controller;
    this.fov_ = 0.0;
    this.width_ = 0;
    this.height_ = 0;
    this.linear_params_ = vec4.create();
    this.view_ = mat4.create();
    this.world_ = mat4.create();
    this.proj_ = mat4.create();
    this.viewproj_ = mat4.create();
    this.bUseOrtho_ = false;

    // Initialize view matrix
    mat4.translate(this.view_, this.view_, [0.0, 0.0, -100.0]); //
  }

  initialized() {
    return this.fov_ > 0.0 && this.width_ > 0 && this.height_ > 0;
  }

  setPerspective(fov, w, h, znear = Camera.kDefaultNear, zfar = Camera.kDefaultFar) {
    console.assert(fov != undefined);
    console.assert(h != undefined);
    console.assert(w != undefined);
    if (fov <= 0.0 || w <= 0 || h <= 0 || zfar <= znear) {
      throw new Error('Invalid perspective parameters.');
    }

    this.fov_ = fov;
    this.width_ = w;
    this.height_ = h;

    const ratio = this.width_ / this.height_;
    mat4.perspective(this.proj_, fov, ratio, znear, zfar);
    this.bUseOrtho_ = false;

    // Linearization parameters
    const A = zfar / (zfar - znear);
    vec4.set(this.linear_params_, znear, zfar, A, -znear * A);
  }

  setDefault() {
    this.setPerspective(Camera.kDefaultFOV, Camera.kDefaultSize, Camera.kDefaultSize, Camera.kDefaultNear, Camera.kDefaultFar);
  }

  update(dt) {
    if (this.controller_ && this.controller_.update) {
      this.controller_.update(dt);
    }
    this.rebuild();
  }

  rebuild(bRetrieveView = true) {
    if (this.controller_ && bRetrieveView) {
      this.controller_.getViewMatrix(this.view_);
    }
    mat4.invert(this.world_, this.view_); // Calculate world matrix
    mat4.multiply(this.viewproj_, this.proj_, this.view_); // View-projection matrix
  }

  setController(controller) {
    this.controller_ = controller;
  }

  fov() {
    return this.fov_;
  }

  width() {
    return this.width_;
  }

  height() {
    return this.height_;
  }

  aspect() {
    return this.width_ / this.height_;
  }

  znear() {
    return this.linear_params_[0];
  }

  zfar() {
    return this.linear_params_[1];
  }

  linearizationParams() {
    return this.linear_params_;
  }

  view() {
    return this.view_;
  }

  world() {
    return this.world_;
  }

  proj() {
    return this.proj_;
  }

  viewproj() {
    return this.viewproj_;
  }

  position() {
    return vec3.fromValues(this.world_[12], this.world_[13], this.world_[14]);
  }

  direction() {
    const direction = vec3.fromValues(
      -this.world_[8],
      -this.world_[9],
      -this.world_[10]
    );
    return vec3.normalize(direction, direction);
  }

  target() {
    if (this.controller_) {
      return this.controller_.target();
    }
    const dir = this.direction();
    const pos = this.position();
    return vec3.add(vec3.create(), pos, vec3.scale(vec3.create(), dir, 3.0));
  }

  isOrtho() {
    return this.bUseOrtho_;
  }
}

// ----------------------------------------------------------------------------

export class DefaultViewController {
  constructor() {
    this.view_ = mat4.create();
    mat4.translate(this.view_, this.view_, [0.0, 0.0, -50.0]);
  }

  update(dt) {
    //
  }

  getViewMatrix(viewMatrix) {
    mat4.copy(viewMatrix, this.view_);
  }

  target() {
    return vec3.fromValues(0.0, 0.0, 0.0);
  }
}

// ----------------------------------------------------------------------------
