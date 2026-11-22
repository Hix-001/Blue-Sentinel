import numpy as np

class AISPreprocessor:
    """
    Standard preprocessor for Blue Sentinel LSTM model.
    Maps raw AIS telemetry into normalized (10, 5) tensor vectors:
    [norm_lat, norm_lon, norm_sog, sin_cog, cos_cog]
    """
    def __init__(self):
        self.lat_min, self.lat_max = -90.0, 90.0
        self.lon_min, self.lon_max = -180.0, 180.0
        self.sog_min, self.sog_max = 0.0, 30.0

    def normalize_lat_lon(self, lat: float, lon: float):
        norm_lat = (lat - self.lat_min) / (self.lat_max - self.lat_min)
        norm_lon = (lon - self.lon_min) / (self.lon_max - self.lon_min)
        return float(np.clip(norm_lat, 0.0, 1.0)), float(np.clip(norm_lon, 0.0, 1.0))

    def denormalize_lat_lon(self, norm_lat: float, norm_lon: float):
        lat = norm_lat * (self.lat_max - self.lat_min) + self.lat_min
        lon = norm_lon * (self.lon_max - self.lon_min) + self.lon_min
        return float(lat), float(lon)

    def normalize_sog(self, sog: float):
        norm_sog = (sog - self.sog_min) / (self.sog_max - self.sog_min)
        return float(np.clip(norm_sog, 0.0, 1.0))

    def encode_cog(self, cog: float):
        rad = np.radians(cog % 360.0)
        return float(np.sin(rad)), float(np.cos(rad))

    def process_point(self, lat: float, lon: float, sog: float, cog: float):
        norm_lat, norm_lon = self.normalize_lat_lon(lat, lon)
        norm_sog = self.normalize_sog(sog)
        sin_cog, cos_cog = self.encode_cog(cog)
        return [norm_lat, norm_lon, norm_sog, sin_cog, cos_cog]

    def process_sequence(self, history_points: list):
        """
        Accepts a list of [lat, lon, sog, cog] tuples/lists of length >= 10.
        Pads or slices to 10 time steps, returns shape (1, 10, 5) float32 numpy array.
        """
        if not history_points:
            history_points = [[10.0, 75.0, 10.0, 0.0]] * 10
            
        pts = history_points[-10:]
        while len(pts) < 10:
            pts.insert(0, pts[0])
            
        vector_seq = [self.process_point(p[0], p[1], p[2], p[3]) for p in pts]
        return np.array([vector_seq], dtype=np.float32)
