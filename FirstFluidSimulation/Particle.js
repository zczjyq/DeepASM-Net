class Particle {
    constructor(position) {
        this.position = position;
        this.prePosition = position;
        this.velocity = Vector2.Zero();
        // 随机颜色
        const seconds = Math.floor(Date.now() / 10); // 以秒为单位
        const hue = seconds % 360; // 色相每360秒循环一次（6分钟一轮）
        this.color = `hsl(${hue}, 80%, 60%)`;
    }
}