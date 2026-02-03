class Playground {
    constructor( ) {
        this.simulation = new Simulation();
        this.mousePos = Vector2.Zero();
    }

    update(dt) {
        this.simulation.update(0.25);
    }

    draw() {
        this.simulation.draw();
    }

    onMouseMove(position) {
        // console.log("Mouse moved to: "+vec.x+", "+vec.y);
        this.mousePos = position;
    }

    onMouseDown(button) {
        if (button == 1) {
            this.simulation.rotate = !this.simulation.rotate;
        }
        if (button == 0) {
            console.log(this.simulation.emitter);
            
            this.simulation.emitter.amount = 8;

        }
        console.log("Mouse button pressed: " + button);
    }

    
    onMouseUp(button) {
        if (button == 0) {
            console.log(this.simulation.emitter);
            
            this.simulation.emitter.amount = 0;

        }
        console.log("Mouse button release: " + button);
    }
}