"""
Готовит датасет для бинарной классификации затопления по снимкам Sentinel-2

Скрипт:
- читает JSON с описанием снимков из SEN12-FLOOD;
- отбирает снимки с полным покрытием данных;
- читает каналы B03 и B08;
- рассчитывает NDWI;
- собирает трехканальное изображение для модели;
- добавляет тайлы Мзымты из SEN-MZYMTA-FLOOD;
- делит данные на train и val;
- сохраняет изображения в формате .npy и таблицы train.csv / val.csv
"""

import json
import re
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from sklearn.model_selection import train_test_split
from tqdm import tqdm


#----------------------------------------------------
# папка с исходными данными
RAW_ROOT = Path("data/raw")

# папка для результата предобработки
OUT_ROOT = Path("data/classification_preprocessed")

# имя json-файла с описанием снимков
JSON_FILENAME = "S2list.json"

# доля данных общего датасета, которая уходит в валидацию
VAL_SIZE = 0.20

# зерно случайности для воспроизводимого деления и балансировки
RANDOM_STATE = 42
#----------------------------------------------------
# малое число для защиты от деления на ноль при расчете индексов
EPS = 1e-6

# размер квадратного тайла, который вырезается из снимков Мзымты
TILE_SIZE = 512

# количество тайлов в одной строке сетки Мзымты
MZYMTA_TILES_PER_ROW = 2

# номера тайлов Мзымты, которые добавляются в итоговый датасет
MZYMTA_TILE_NUMBERS = [2, 4, 5]
#----------------------------------------------------
# порядок колонок в выходных csv-файлах
OUTPUT_COLUMNS = [
    "sample_id",
    "folder",
    "date",
    "filename",
    "flooding",
    "label",
    "source",
    "tile_number",
]
#----------------------------------------------------
# вручную заданные снимки Мзымты и их метки затопления
MZYMTA_SAMPLES = [
    {
        "folder": "data/mzymta/2021-07-01",
        "flooding": False,
    },
    {
        "folder": "data/mzymta/2021-07-09",
        "flooding": True,
    },
    {
        "folder": "data/mzymta/2021-07-19",
        "flooding": False,
    },
    {
        "folder": "data/mzymta/2021-07-29",
        "flooding": True,
    },
    {
        "folder": "data/mzymta/2021-12-03",
        "flooding": False,
    },
    {
        "folder": "data/mzymta/2021-12-13",
        "flooding": True,
    },
    {
        "folder": "data/mzymta/2023-07-06",
        "flooding": False,
    },
    {
        "folder": "data/mzymta/2023-07-14",
        "flooding": True,
    },
]
#----------------------------------------------------
# даты Мзымты, которые принудительно относятся к валидационной выборке
# это нужно, чтобы снимки с одного дня не попали и в обучение, и в валидацию
MZYMTA_VAL_DATES = {
    "2021-07-19",
    "2021-07-29",
}
#----------------------------------------------------



#----------------------------------------------------
def clean_output_folder(out_root: Path) -> None:
    """
    Очищает папку результата перед новой предобработкой

    Args:
        out_root (Path): Папка, куда будут сохранены .npy-файлы, csv-файлы и расширенный JSON

    Raises:
        ValueError: Возникает, если передан неверный путь для очистки
    """

    # защита от случайного удаления текущей или корневой директории
    if str(out_root).strip() in ["", ".", "/", "\\"]:
        raise ValueError(f"Нельзя очищать неверный путь: {out_root}")

    # удаляем старый результат, чтобы не смешать файлы разных запусков
    if out_root.exists():
        shutil.rmtree(out_root)

    # создаем папку для подготовленных изображений
    (out_root / "images").mkdir(parents=True, exist_ok=True)


#----------------------------------------------------
def find_json_path(raw_root: Path) -> Path:
    """
    Находит исходный JSON-файл S2list.json с описанием SEN12-FLOOD датасета

    Args:
        raw_root (Path): Корневая папка исходного датасета

    Returns:
        Path: Путь к найденному JSON-файлу

    Raises:
        FileNotFoundError: Возникает, если JSON-файл не найден
    """

    # в датасете используется фиксированное имя json-файла
    json_path = raw_root / JSON_FILENAME

    if not json_path.exists():
        raise FileNotFoundError(f"JSON-файл {JSON_FILENAME} не найден в папке: {raw_root}")

    return json_path

#----------------------------------------------------
def read_json(json_path: Path) -> dict:
    """
    Читает JSON-файл с описанием снимков

    Args:
        json_path (Path): Путь к исходному JSON-файлу

    Returns:
        dict: Содержимое JSON-файла в виде словаря
    """

    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


#----------------------------------------------------
def save_json(data: dict, json_path: Path) -> None:
    """
    Сохраняет словарь в JSON-файл

    Args:
        data (dict): Данные, которые нужно сохранить
        json_path (Path): Путь к выходному JSON-файлу
    """

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


#----------------------------------------------------
def read_band(path: Path) -> np.ndarray:
    """
    Читает один спектральный канал без изменения размера

    Args:
        path (Path): Путь к файлу спектрального канала

    Returns:
        np.ndarray: Двумерный массив значений канала в формате float32
    """

    with rasterio.open(path) as dataset:
        band = dataset.read(1).astype(np.float32)

    return band


#----------------------------------------------------
def read_band_resampled(path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Читает спектральный канал и приводит его к заданному размеру

    Args:
        path (Path): Путь к файлу спектрального канала
        target_shape (tuple[int, int]): Целевой размер массива в формате (height, width)

    Returns:
        np.ndarray: Двумерный массив канала после билинейного ресемплинга
    """

    # ресемплинг нужен, если каналы Sentinel-2 имеют разное пространственное разрешение
    with rasterio.open(path) as dataset:
        band = dataset.read(
            1,
            out_shape=target_shape,
            resampling=Resampling.bilinear,
        ).astype(np.float32)

    return band


#----------------------------------------------------
def normalize_percentile(channel: np.ndarray) -> np.ndarray:
    """
    Нормализует канал по 2 и 98 процентилям

    Args:
        channel (np.ndarray): Двумерный массив значений спектрального канала

    Returns:
        np.ndarray: Нормализованный канал со значениями в диапазоне от 0 до 1
    """

    # берем только конечные значения, чтобы NaN и inf не ломали расчет процентилей
    valid = channel[np.isfinite(channel)]

    if valid.size == 0:
        return np.zeros_like(channel, dtype=np.float32)

    # отсекаем крайние 2 процента снизу и сверху, чтобы уменьшить влияние выбросов
    low = np.percentile(valid, 2)
    high = np.percentile(valid, 98)

    # если канал почти константный, нормализация не имеет смысла
    if high - low < EPS:
        return np.zeros_like(channel, dtype=np.float32)

    result = (channel - low) / (high - low)
    result = np.clip(result, 0.0, 1.0)

    return result.astype(np.float32)


#----------------------------------------------------
def calculate_ndwi(b03: np.ndarray, b08: np.ndarray) -> np.ndarray:
    """
    Рассчитывает индекс NDWI по зеленому и ближнему инфракрасному каналам

    Args:
        b03 (np.ndarray): Зеленый канал Sentinel-2
        b08 (np.ndarray): Ближний инфракрасный канал Sentinel-2

    Returns:
        np.ndarray: Массив NDWI со значениями в диапазоне от -1 до 1
    """

    # NDWI усиливает водную поверхность за счет разницы между green и NIR
    ndwi = (b03 - b08) / (b03 + b08 + EPS)
    ndwi = np.clip(ndwi, -1.0, 1.0)

    return ndwi.astype(np.float32)


#----------------------------------------------------
def build_model_input(b03: np.ndarray, b08: np.ndarray, ndwi: np.ndarray) -> np.ndarray:
    """
    Собирает трехканальное изображение для входа модели классификации

    Args:
        b03 (np.ndarray): Зеленый канал Sentinel-2
        b08 (np.ndarray): Ближний инфракрасный канал Sentinel-2
        ndwi (np.ndarray): Индекс NDWI, рассчитанный по каналам B03 и B08

    Returns:
        np.ndarray: Массив изображения размера (3, height, width) в формате float32
    """

    # приводим исходные каналы к диапазону 0..1
    green_norm = normalize_percentile(b03)
    nir_norm = normalize_percentile(b08)

    # переводим NDWI из диапазона -1..1 в диапазон 0..1
    ndwi_norm = ((ndwi + 1.0) / 2.0).astype(np.float32)

    # модель получает три признака: green, NIR и NDWI
    image = np.stack(
        [
            green_norm,
            nir_norm,
            ndwi_norm,
        ],
        axis=0,
    )

    return image.astype(np.float32)


#----------------------------------------------------
def find_band_path(folder_path: Path, filename: str, band_name: str) -> Path | None:
    """
    Ищет файл нужного спектрального канала внутри папки снимка

    Args:
        folder_path (Path): Папка, где лежат файлы снимка
        filename (str): Базовое имя снимка из JSON или сформированное по дате
        band_name (str): Имя канала, например B03 или B08

    Returns:
        Path | None: Путь к найденному файлу канала или None, если файл не найден
    """

    extensions = [".tif", ".tiff", ".jp2", ".png"]

    # сначала проверяем самые частые варианты имени файла без рекурсивного поиска
    direct_variants = []

    for extension in extensions:
        direct_variants.append(folder_path / f"{filename}_{band_name}{extension}")
        direct_variants.append(folder_path / f"{filename}-{band_name}{extension}")
        direct_variants.append(folder_path / f"{filename}.{band_name}{extension}")

    for path in direct_variants:
        if path.exists():
            return path

    # если прямое имя не подошло, ищем файл рекурсивно по имени снимка и названию канала
    band_pattern = re.compile(rf"(^|[_\-.]){re.escape(band_name)}($|[_\-.])", re.IGNORECASE)
    filename_lower = filename.lower()

    candidates = []

    for path in folder_path.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in extensions:
            continue

        name = path.stem
        name_lower = name.lower()

        if filename_lower not in name_lower:
            continue

        if band_pattern.search(name):
            candidates.append(path)

    if len(candidates) == 0:
        return None

    # выбираем самый короткий путь как наиболее прямое совпадение
    candidates = sorted(candidates, key=lambda item: len(str(item)))

    return candidates[0]


#----------------------------------------------------
def get_filename(item: dict) -> str | None:
    """
    Получает имя снимка из JSON-записи

    Args:
        item (dict): JSON-запись одного снимка

    Returns:
        str | None: Имя снимка или None, если нельзя получить имя
    """

    filename = item.get("filename")

    if filename:
        return filename

    # если имени нет, пробуем восстановить его по дате
    date = item.get("date")

    if date:
        return f"S2_{date}"

    return None


#----------------------------------------------------
def extract_date_from_folder(folder: str) -> str:
    """
    Извлекает дату из пути к папке

    Args:
        folder (str): Строка с путем к папке снимка

    Returns:
        str: Дата в формате YYYY-MM-DD

    Raises:
        ValueError: Возникает, если в пути не найдена дата
    """

    match = re.search(r"\d{4}-\d{2}-\d{2}", folder)

    if not match:
        raise ValueError(f"Не удалось извлечь дату из пути: {folder}")

    return match.group(0)


#----------------------------------------------------
def safe_name(text: str) -> str:
    """
    Делает строку безопасной для имени npy-файла

    Args:
        text (str): Исходная строка с путем, именем снимка или их комбинацией

    Returns:
        str: Строка без опасных символов для имени файла
    """

    # приводим разделители путей к одному виду
    text = text.replace("\\", "/")

    # заменяем все неподходящие символы на нижнее подчеркивание
    text = re.sub(r"[^A-Za-z0-9А-Яа-я_.-]+", "_", text)
    text = text.strip("_")

    return text


#----------------------------------------------------
def make_sample_id(folder: str, filename: str, tile_number: int | None = None) -> str:
    """
    Создает уникальный идентификатор снимка или тайла

    Args:
        folder (str): Папка исходного снимка
        filename (str): Имя исходного снимка
        tile_number (int | None): Номер тайла, если идентификатор создается для тайла

    Returns:
        str: Уникальный идентификатор для сохранения .npy-файла и записи в csv
    """

    sample_id = safe_name(f"{folder}_{filename}")

    # для тайлов добавляем номер, чтобы не перезаписать файлы одной даты
    if tile_number is not None:
        sample_id = f"{sample_id}_tile_{tile_number:02d}"

    return sample_id


#----------------------------------------------------
def build_image_from_bands(folder_path: Path, filename: str) -> tuple[np.ndarray | None, dict | None]:
    """
    Читает каналы B03 и B08, рассчитывает NDWI и собирает входное изображение модели

    Args:
        folder_path (Path): Папка, где лежат файлы снимка
        filename (str): Базовое имя снимка

    Returns:
        tuple[np.ndarray | None, dict | None]: Изображение модели и ошибка. Если обработка успешна,
        возвращается (image, None). Если обработка неуспешна, возвращается (None, error)
    """

    # ищем два канала, необходимые для формирования входа модели
    b03_path = find_band_path(folder_path, filename, "B03")
    b08_path = find_band_path(folder_path, filename, "B08")

    missing_bands = []

    if b03_path is None:
        missing_bands.append("B03")

    if b08_path is None:
        missing_bands.append("B08")

    # если хотя бы одного канала нет, снимок нельзя использовать
    if missing_bands:
        return None, {
            "reason": "missing_bands:" + ",".join(missing_bands),
        }

    try:
        # B03 читается как базовый канал, а B08 приводится к его размеру
        b03 = read_band(b03_path)
        b08 = read_band_resampled(b08_path, target_shape=b03.shape)
    except Exception as exception:
        return None, {
            "reason": f"read_error:{exception}",
        }

    # после ресемплинга размеры должны полностью совпасть
    if b03.shape != b08.shape:
        return None, {
            "reason": f"shape_mismatch:B03={b03.shape},B08={b08.shape}",
        }

    # считаем индекс и собираем трехканальное изображение
    ndwi = calculate_ndwi(b03, b08)
    image = build_model_input(b03, b08, ndwi)

    return image, None


#----------------------------------------------------
def process_regular_sample(
    raw_root: Path,
    out_root: Path,
    folder: str,
    item: dict,
) -> tuple[dict | None, dict | None]:
    """
    Обрабатывает один снимок из общего датасета

    Args:
        raw_root (Path): Корневая папка общего исходного датасета
        out_root (Path): Папка для сохранения результата предобработки
        folder (str): Относительный путь к папке снимка из JSON
        item (dict): JSON-запись одного снимка

    Returns:
        tuple[dict | None, dict | None]: Строка для итоговой таблицы и строка с причиной пропуска
        При успешной обработке возвращается (row, None), при ошибке - (None, skipped)
    """

    # получаем имя снимка из filename или восстанавливаем по дате
    filename = get_filename(item)

    if not filename:
        return None, {
            "folder": folder,
            "date": item.get("date"),
            "filename": "",
            "reason": "missing_filename_and_date",
        }

    folder_path = raw_root / folder

    # строим вход модели из каналов Sentinel-2
    image, error = build_image_from_bands(
        folder_path=folder_path,
        filename=filename,
    )

    if error is not None:
        return None, {
            "folder": folder,
            "date": item.get("date"),
            "filename": filename,
            "reason": error["reason"],
        }

    # переводим исходную булеву метку в числовой класс
    flooding = bool(item.get("FLOODING"))
    label = 1 if flooding else 0

    # сохраняем подготовленный массив изображения
    sample_id = make_sample_id(folder, filename)
    image_path = out_root / "images" / f"{sample_id}.npy"

    np.save(image_path, image)

    # формируем запись для train/val csv
    row = {
        "sample_id": sample_id,
        "folder": folder,
        "date": item.get("date"),
        "filename": filename,
        "flooding": flooding,
        "label": label,
        "source": "raw",
        "tile_number": "",
    }

    return row, None


#----------------------------------------------------
def collect_regular_samples(
    raw_root: Path,
    out_root: Path,
    json_data: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Собирает валидные снимки из общего датасета

    Args:
        raw_root (Path): Корневая папка общего исходного датасета
        out_root (Path): Папка для сохранения подготовленных .npy-файлов
        json_data (dict): Содержимое исходного JSON-файла

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Таблица валидных снимков и таблица пропущенных снимков
    """

    rows = []
    skipped_rows = []

    # верхний уровень JSON - это папки со снимками
    for folder, sequence in tqdm(json_data.items(), desc="Processing raw dataset"):
        if not isinstance(sequence, dict):
            continue

        # внутри каждой папки обрабатываются только числовые ключи снимков
        for key, item in sequence.items():
            if not str(key).isdigit():
                continue

            if not isinstance(item, dict):
                continue

            # снимки без полного покрытия всегда исключаются из датасета
            if item.get("FULL-DATA-COVERAGE") is not True:
                skipped_rows.append(
                    {
                        "folder": folder,
                        "date": item.get("date"),
                        "filename": item.get("filename"),
                        "reason": "not_full_data_coverage",
                    }
                )
                continue

            row, skipped = process_regular_sample(
                raw_root=raw_root,
                out_root=out_root,
                folder=folder,
                item=item,
            )

            if row is not None:
                rows.append(row)

            if skipped is not None:
                skipped_rows.append(skipped)

    dataframe = pd.DataFrame(rows)
    skipped_dataframe = pd.DataFrame(skipped_rows)

    return dataframe, skipped_dataframe


#----------------------------------------------------
def get_tile_bounds(tile_number: int) -> tuple[int, int, int, int]:
    """
    Возвращает координаты тайла 512 на 512 по сетке 1 2 / 3 4 / 5 6 / 7 8

    Args:
        tile_number (int): Номер тайла, начиная с 1

    Returns:
        tuple[int, int, int, int]: Координаты тайла в формате (y0, y1, x0, x1)
    """

    # переводим номер тайла в индекс, начиная с нуля
    tile_index = tile_number - 1

    # определяем строку и столбец тайла в сетке
    row = tile_index // MZYMTA_TILES_PER_ROW
    col = tile_index % MZYMTA_TILES_PER_ROW

    # рассчитываем границы вырезки в пикселях
    y0 = row * TILE_SIZE
    x0 = col * TILE_SIZE
    y1 = y0 + TILE_SIZE
    x1 = x0 + TILE_SIZE

    return y0, y1, x0, x1


#----------------------------------------------------
def cut_selected_tiles(image: np.ndarray) -> tuple[list[tuple[int, np.ndarray]], list[dict]]:
    """
    Вырезает из снимка Мзымты только выбранные тайлы

    Args:
        image (np.ndarray): Трехканальное изображение размера (3, height, width)

    Returns:
        tuple[list[tuple[int, np.ndarray]], list[dict]]: Список валидных тайлов и список причин пропуска тайлов
    """

    tiles = []
    skipped_rows = []

    height = image.shape[1]
    width = image.shape[2]

    for tile_number in MZYMTA_TILE_NUMBERS:
        y0, y1, x0, x1 = get_tile_bounds(tile_number)

        # проверяем, помещается ли тайл в исходный снимок
        if y1 > height or x1 > width:
            skipped_rows.append(
                {
                    "tile_number": tile_number,
                    "reason": f"tile_out_of_bounds:tile={tile_number},image_shape={image.shape}",
                }
            )
            continue

        tile = image[:, y0:y1, x0:x1]

        # проверяем итоговую форму, чтобы в модель не попал поврежденный тайл
        if tile.shape != (3, TILE_SIZE, TILE_SIZE):
            skipped_rows.append(
                {
                    "tile_number": tile_number,
                    "reason": f"wrong_tile_shape:tile={tile_number},tile_shape={tile.shape}",
                }
            )
            continue

        tiles.append((tile_number, tile))

    return tiles, skipped_rows


#----------------------------------------------------
def process_mzymta_sample(
    out_root: Path,
    sample: dict,
) -> tuple[list[dict], list[dict]]:
    """
    Обрабатывает один снимок Мзымты и сохраняет выбранные тайлы

    Args:
        out_root (Path): Папка для сохранения результата предобработки
        sample (dict): Описание снимка Мзымты с папкой и меткой затопления

    Returns:
        tuple[list[dict], list[dict]]: Список строк для итоговой таблицы и список причин пропуска
    """

    folder = sample["folder"]
    folder_path = Path(folder)

    # имя снимка Мзымты восстанавливается по дате в пути
    date = extract_date_from_folder(folder)
    filename = f"S2_{date}"

    rows = []
    skipped_rows = []

    # собираем трехканальное изображение из спектральных каналов
    image, error = build_image_from_bands(
        folder_path=folder_path,
        filename=filename,
    )

    if error is not None:
        skipped_rows.append(
            {
                "folder": folder,
                "date": date,
                "filename": filename,
                "reason": error["reason"],
            }
        )
        return rows, skipped_rows

    # режем большой снимок на заранее выбранные тайлы
    tiles, skipped_tiles = cut_selected_tiles(image)

    for skipped_tile in skipped_tiles:
        skipped_rows.append(
            {
                "folder": folder,
                "date": date,
                "filename": filename,
                "reason": skipped_tile["reason"],
            }
        )

    # метка всего снимка переносится на каждый выбранный тайл
    flooding = bool(sample["flooding"])
    label = 1 if flooding else 0

    for tile_number, tile in tiles:
        sample_id = make_sample_id(folder, filename, tile_number=tile_number)
        image_path = out_root / "images" / f"{sample_id}.npy"

        # каждый тайл сохраняется как отдельный npy-файл
        np.save(image_path, tile)

        row = {
            "sample_id": sample_id,
            "folder": folder,
            "date": date,
            "filename": filename,
            "flooding": flooding,
            "label": label,
            "source": "mzymta",
            "tile_number": tile_number,
        }

        rows.append(row)

    return rows, skipped_rows


#----------------------------------------------------
def collect_mzymta_samples(out_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Собирает тайлы Мзымты для добавления в общий датасет

    Args:
        out_root (Path): Папка для сохранения подготовленных .npy-файлов

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Таблица валидных тайлов и таблица пропущенных тайлов
    """

    rows = []
    skipped_rows = []

    for sample in tqdm(MZYMTA_SAMPLES, desc="Processing Mzymta tiles"):
        sample_rows, sample_skipped_rows = process_mzymta_sample(
            out_root=out_root,
            sample=sample,
        )

        rows.extend(sample_rows)
        skipped_rows.extend(sample_skipped_rows)

    dataframe = pd.DataFrame(rows)
    skipped_dataframe = pd.DataFrame(skipped_rows)

    return dataframe, skipped_dataframe


#----------------------------------------------------
def get_mzymta_split(date: str) -> str:
    """
    Определяет часть выборки для снимка Мзымты по дате

    Args:
        date (str): Дата снимка Мзымты в формате YYYY-MM-DD

    Returns:
        str: Строка "val", если дата входит в MZYMTA_VAL_DATES, иначе "train"
    """

    if date in MZYMTA_VAL_DATES:
        return "val"

    return "train"


#----------------------------------------------------
def split_mzymta_dataframe(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Делит Мзымту на train и val по датам

    Args:
        dataframe (pd.DataFrame): Таблица подготовленных тайлов Мзымты

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Обучающая и валидационная части Мзымты

    Raises:
        RuntimeError: Возникает, если train или val пустые либо не содержат оба класса
    """

    if len(dataframe) == 0:
        empty_dataframe = pd.DataFrame(columns=dataframe.columns)
        return empty_dataframe, empty_dataframe

    # тайлы одной даты не должны попадать одновременно в train и val
    val_df = dataframe[dataframe["date"].isin(MZYMTA_VAL_DATES)].copy()
    train_df = dataframe[~dataframe["date"].isin(MZYMTA_VAL_DATES)].copy()

    # проверки ниже защищают от некорректного эксперимента с пустой или одноклассовой выборкой
    if len(val_df) == 0:
        raise RuntimeError("Валидационная часть Мзымты пустая. Проверь MZYMTA_VAL_DATES.")

    if len(train_df) == 0:
        raise RuntimeError("Обучающая часть Мзымты пустая. Проверь MZYMTA_VAL_DATES.")

    if val_df["label"].nunique() < 2:
        raise RuntimeError("Валидация Мзымты должна содержать оба класса: flood и no_flood.")

    if train_df["label"].nunique() < 2:
        raise RuntimeError("Обучение Мзымты должно содержать оба класса: flood и no_flood.")

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    return train_df, val_df


#----------------------------------------------------
def concat_dataframes(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Объединяет непустые DataFrame в одну таблицу

    Args:
        dataframes (list[pd.DataFrame]): Список таблиц для объединения

    Returns:
        pd.DataFrame: Общая таблица или пустой DataFrame, если все входные таблицы пустые
    """

    non_empty_dataframes = []

    # пустые таблицы пропускаются, чтобы не ломать pd.concat
    for dataframe in dataframes:
        if len(dataframe) > 0:
            non_empty_dataframes.append(dataframe)

    if len(non_empty_dataframes) == 0:
        return pd.DataFrame()

    return pd.concat(non_empty_dataframes, ignore_index=True)


#----------------------------------------------------
def extend_json_with_mzymta(json_data: dict) -> dict:
    """
    Создает копию исходного JSON и добавляет в нее снимки Мзымты

    Args:
        json_data (dict): Исходный JSON общего датасета

    Returns:
        dict: Расширенный JSON с дополнительными записями Мзымты
    """

    # исходный JSON не меняется напрямую, чтобы не испортить входные данные
    extended_data = deepcopy(json_data)

    for sample in MZYMTA_SAMPLES:
        folder = sample["folder"].replace("\\", "/")
        date = extract_date_from_folder(folder)
        filename = f"S2_{date}"

        # если папки Мзымты нет в исходном JSON, создаем новую секцию
        if folder not in extended_data or not isinstance(extended_data[folder], dict):
            extended_data[folder] = {}

        sequence = extended_data[folder]

        # ищем последний числовой ключ, чтобы добавить новую запись без перезаписи старых
        numeric_keys = []

        for key in sequence.keys():
            if str(key).isdigit():
                numeric_keys.append(int(key))

        next_key = str(max(numeric_keys) + 1) if numeric_keys else "1"

        # добавляем описание снимка Мзымты в том же стиле, что и общий датасет
        sequence[next_key] = {
            "date": date,
            "filename": filename,
            "FLOODING": bool(sample["flooding"]),
            "FULL-DATA-COVERAGE": True,
            "SOURCE": "mzymta",
            "TILE-SIZE": TILE_SIZE,
            "TILE-NUMBERS": MZYMTA_TILE_NUMBERS,
        }

    return extended_data


#----------------------------------------------------
def balance_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Балансирует классы за счет уменьшения большего класса

    Args:
        dataframe (pd.DataFrame): Таблица снимков с колонкой label

    Returns:
        pd.DataFrame: Сбалансированная таблица с одинаковым числом объектов двух классов

    Raises:
        RuntimeError: Возникает, если в таблице нет одного из классов
    """

    class_counts = dataframe["label"].value_counts().sort_index()

    if 0 not in class_counts.index or 1 not in class_counts.index:
        raise RuntimeError(f"Нельзя сбалансировать датасет: найдены классы {class_counts.to_dict()}")

    # размер меньшего класса становится целевым размером для каждого класса
    min_count = int(class_counts.min())

    balanced_parts = []

    for label in [0, 1]:
        # из большего класса случайно выбирается столько же объектов, сколько есть в меньшем
        part = dataframe[dataframe["label"] == label].sample(
            n=min_count,
            random_state=RANDOM_STATE,
        )
        balanced_parts.append(part)

    balanced_dataframe = pd.concat(balanced_parts, ignore_index=True)

    # перемешиваем строки после объединения классов
    balanced_dataframe = balanced_dataframe.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return balanced_dataframe


#----------------------------------------------------
def can_use_stratify(dataframe: pd.DataFrame) -> bool:
    """
    Проверяет, можно ли использовать стратифицированное деление train/val

    Args:
        dataframe (pd.DataFrame): Таблица снимков с колонкой label

    Returns:
        bool: True, если stratify можно использовать, иначе False
    """

    class_counts = dataframe["label"].value_counts()

    # stratify невозможен, если есть только один класс
    if len(class_counts) < 2:
        return False

    # в каждом классе должно быть хотя бы два объекта, чтобы разделить его на train и val
    if class_counts.min() < 2:
        return False

    val_count = int(np.ceil(len(dataframe) * VAL_SIZE))
    train_count = len(dataframe) - val_count

    # train и val должны вмещать оба класса
    if val_count < len(class_counts):
        return False

    if train_count < len(class_counts):
        return False

    return True


#----------------------------------------------------
def split_train_val_dataframe(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Делит общий датасет на train и val в пропорции 80/20

    Args:
        dataframe (pd.DataFrame): Таблица подготовленных снимков

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Обучающая и валидационная таблицы

    Raises:
        ValueError: Возникает, если VAL_SIZE находится вне диапазона от 0 до 1
    """

    if VAL_SIZE <= 0 or VAL_SIZE >= 1:
        raise ValueError("VAL_SIZE должен быть больше 0 и меньше 1.")

    # stratify сохраняет соотношение классов, но используется только когда это безопасно
    stratify_column = dataframe["label"] if can_use_stratify(dataframe) else None

    train_df, val_df = train_test_split(
        dataframe,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_column,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    return train_df, val_df


#----------------------------------------------------
def prepare_output_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Оставляет в выходной таблице только нужные колонки

    Args:
        dataframe (pd.DataFrame): Таблица train или val перед сохранением

    Returns:
        pd.DataFrame: Таблица с колонками в фиксированном порядке
    """

    # если какой-то колонки нет, создаем ее пустой, чтобы csv всегда имел одинаковую структуру
    for column in OUTPUT_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = ""

    dataframe = dataframe[OUTPUT_COLUMNS].copy()

    return dataframe


#----------------------------------------------------
def print_class_counts(title: str, dataframe: pd.DataFrame) -> None:
    """
    Печатает количество объектов по классам

    Args:
        title (str): Заголовок блока вывода
        dataframe (pd.DataFrame): Таблица с колонкой label
    """

    print(title)

    if len(dataframe) == 0:
        print("empty")
        print()
        return

    print(dataframe["label"].value_counts().sort_index().rename(index={0: "no_flood", 1: "flood"}))
    print()


#----------------------------------------------------
def print_skipped(title: str, dataframe: pd.DataFrame) -> None:
    """
    Печатает причины пропуска снимков

    Args:
        title (str): Заголовок блока вывода
        dataframe (pd.DataFrame): Таблица с колонкой reason
    """

    print(title)

    if len(dataframe) == 0:
        print("Пропущенных снимков нет.")
        print()
        return

    # показываем самые частые причины, чтобы быстро понять проблемы во входных данных
    print(dataframe["reason"].value_counts().head(30))
    print()


#----------------------------------------------------
def main() -> None:
    """
    Запускает полную предобработку датасета

    Последовательно выполняет:
    - использует фиксированные пути и настройки
    - ищет и читает JSON
    - очищает папку результата
    - обрабатывает общий датасет
    - добавляет тайлы Мзымты
    - балансирует классы
    - делит данные на train и val в пропорции 80/20
    - сохраняет csv, npy и расширенный JSON
    """

    raw_root = RAW_ROOT
    out_root = OUT_ROOT

    # проверяем наличие папки с исходными данными до удаления результата
    if not raw_root.exists():
        raise FileNotFoundError(f"Папка raw-root не найдена: {raw_root}")

    # находим и читаем исходное описание датасета
    json_path = find_json_path(raw_root)
    json_data = read_json(json_path)

    # очищаем предыдущую предобработку и создаем структуру выхода
    clean_output_folder(out_root)

    # обрабатываем основной датасет Sentinel-2 из JSON
    regular_dataframe, regular_skipped = collect_regular_samples(
        raw_root=raw_root,
        out_root=out_root,
        json_data=json_data,
    )

    # добавляем вручную заданные тайлы Мзымты
    mzymta_dataframe, mzymta_skipped = collect_mzymta_samples(
        out_root=out_root,
    )

    if len(regular_dataframe) == 0 and len(mzymta_dataframe) == 0:
        raise RuntimeError("Не найдено ни одного валидного снимка. Проверяй raw, json и папки Мзымты.")

    # балансируем общий датасет, чтобы классы flood / no_flood были равны
    if len(regular_dataframe) > 0:
        regular_dataframe = balance_dataframe(
            dataframe=regular_dataframe,
        )

    raw_train_df = pd.DataFrame(columns=regular_dataframe.columns)
    raw_val_df = pd.DataFrame(columns=regular_dataframe.columns)

    # общий датасет делится случайно в пропорции 80/20, при возможности со stratify
    if len(regular_dataframe) > 0:
        raw_train_df, raw_val_df = split_train_val_dataframe(
            dataframe=regular_dataframe,
        )

    # Мзымта делится строго по датам, чтобы не было утечки между тайлами одной даты
    mzymta_train_df, mzymta_val_df = split_mzymta_dataframe(
        dataframe=mzymta_dataframe,
    )

    # объединяем train из общего датасета и train из Мзымты
    train_df = concat_dataframes(
        [
            raw_train_df,
            mzymta_train_df,
        ]
    )

    # объединяем val из общего датасета и val из Мзымты
    val_df = concat_dataframes(
        [
            raw_val_df,
            mzymta_val_df,
        ]
    )

    # перемешиваем итоговые таблицы, чтобы объекты разных источников не шли блоками
    train_df = train_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    # общая таблица нужна только для печати статистики
    dataframe = concat_dataframes(
        [
            train_df,
            val_df,
        ]
    )

    # приводим csv к фиксированному набору колонок
    train_df = prepare_output_dataframe(train_df)
    val_df = prepare_output_dataframe(val_df)

    # сохраняем итоговые таблицы разбиения
    train_df.to_csv(out_root / "train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(out_root / "val.csv", index=False, encoding="utf-8-sig")

    # сохраняем JSON, где общий датасет дополнен снимками Мзымты
    extended_json = extend_json_with_mzymta(json_data)
    save_json(extended_json, out_root / "S2list_extended.json")

    # объединяем причины пропуска из обоих источников
    skipped_dataframe = pd.concat(
        [
            regular_skipped,
            mzymta_skipped,
        ],
        ignore_index=True,
    )

    print()
    print(f"Исходный JSON: {json_path}")
    print(f"Результат: {out_root}")
    print(f"Файлы на выходе: images/*.npy, train.csv, val.csv, S2list_extended.json")
    print()

    print_class_counts("Общий датасет:", dataframe)
    print_class_counts("Train:", train_df)
    print_class_counts("Val:", val_df)

    print(f"Всего снимков/тайлов: {len(dataframe)}")
    print(f"Train: {len(train_df)}")
    print(f"Val: {len(val_df)}")
    print()

    print_skipped("Причины пропуска:", skipped_dataframe)

#----------------------------------------------------
if __name__ == "__main__":
    main()
