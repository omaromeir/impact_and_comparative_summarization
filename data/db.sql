CREATE TABLE public.movies (
    id integer NOT NULL,
    title text,
    original_language character(2),
    runtime character varying(10),
    revenue numeric,
    release_date character varying(30),
    production_country character varying(100),
    dir_id integer,
    vote_average real,
    vote_count integer,
    production_company text
);

COPY movies
FROM '/Users/omaromeir/Documents/GitHub/agg_query_refinement/data/movies_clean.tsv'
DELIMITER '|'
CSV HEADER;


CREATE TABLE public.director (
    id integer NOT NULL,
    name text,
    gender integer
);

COPY director
FROM '/Users/omaromeir/Documents/GitHub/agg_query_refinement/data/directors.tsv'
DELIMITER '|'
CSV HEADER;

CREATE TABLE public.genre (
    movie_id integer NOT NULL,
    genre character varying(30)
);

COPY genre
FROM '/Users/omaromeir/Documents/GitHub/agg_query_refinement/data/genres.tsv'
DELIMITER '|'
CSV HEADER;
