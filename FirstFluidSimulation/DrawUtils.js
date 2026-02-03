class DrawUtils {
    static drawPoint(position, radius, color) {
        ctx.beginPath();
        ctx.arc(position.x, position.y, radius, 0, Math.PI * 2, true);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.closePath();
    }

    static strokePoint(position, radius, color) {
        ctx.beginPath();
        ctx.arc(position.x, position.y, radius, 0, Math.PI * 2, true);
        ctx.strokeStyle = color;
        ctx.stroke();
        ctx.closePath();
    }

    static drawLine(startPosition, endPosition, color, lineThiknes = 1) {
        ctx.beginPath();
        ctx.lineWidth = lineThiknes;
        ctx.moveTo(startPosition.x, startPosition.y);
        ctx.lineTo(endPosition.x, endPosition.y);
        ctx.strokeStyle = color;
        ctx.stroke();
        ctx.lineWidth = 1;
        ctx.closePath();
    }

    static drawRect(start, size, color) {
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.rect(start.x, start.y, size.x, size.y);
        ctx.stroke();
        ctx.closePath();
    }

    static drawText(position, size, color, text) {
        ctx.font = size + "px Arial";
        ctx.fillStyle = color;
        ctx.fillText(text, position.x, position.y);
    }
}